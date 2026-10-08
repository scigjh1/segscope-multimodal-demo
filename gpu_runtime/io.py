from dataclasses import dataclass
from pathlib import Path
import io
import zipfile
import nibabel as nib
import numpy as np
from PIL import Image


@dataclass
class Volume:
    array: np.ndarray
    affine: np.ndarray
    spacing: tuple
    units: str
    source: str


def load_volume(path):
    path = Path(path)
    if path.stat().st_size > 128*1024**2:
        raise ValueError("Input file exceeds 128 MiB")
    lower = path.name.lower()
    if lower.endswith(('.nii', '.nii.gz')):
        image = nib.load(path)
        if len(image.shape) != 3 or np.prod(image.shape) > 128**3*128:
            raise ValueError("Expected bounded 3D NIfTI")
        array = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0)
        units = image.header.get_xyzt_units()[0]
        spacing = tuple(float(x) for x in image.header.get_zooms()[:3][::-1])
        affine = image.affine.copy()
        if units == 'meter':
            spacing = tuple(x*1000 for x in spacing)
            affine[:3] *= 1000
            units = 'mm'
        elif units == 'micron':
            spacing = tuple(x/1000 for x in spacing)
            affine[:3] /= 1000
            units = 'mm'
        return Volume(array, affine, spacing, units if units == 'mm' else 'voxel', 'NIfTI')
    if lower.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(path)
        if image.width*image.height > 16_000_000:
            raise ValueError("Image is too large")
        array = np.asarray(image.convert('L'), dtype=np.float32)[None]/255
        return Volume(array, np.eye(4), (1., 1., 1.), 'voxel', 'PNG/JPEG')
    if lower.endswith(('.dcm', '.dicom', '.zip')):
        import pydicom
        if lower.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                entries = [x for x in archive.infolist() if not x.is_dir()]
                if len(entries) > 4096 or sum(x.file_size for x in entries) > 512*1024**2:
                    raise ValueError("DICOM archive exceeds bounds")
                datasets = [pydicom.dcmread(io.BytesIO(archive.read(x))) for x in entries]
        else:
            datasets = [pydicom.dcmread(path)]
        if any(int(getattr(x, 'NumberOfFrames', 1)) != 1 for x in datasets):
            raise ValueError("Enhanced/multiframe DICOM requires a specialized loader")
        if len({str(x.SeriesInstanceUID) for x in datasets}) != 1:
            raise ValueError("Upload one DICOM series per case")
        orient = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        if any(not np.allclose(x.ImageOrientationPatient, orient, atol=1e-4) for x in datasets):
            raise ValueError("Inconsistent DICOM orientation")
        datasets.sort(key=lambda x: np.dot(x.ImagePositionPatient, normal))
        positions = [float(np.dot(x.ImagePositionPatient, normal)) for x in datasets]
        dz = float(getattr(datasets[0], 'SliceThickness', 1.))
        if len(positions) > 1:
            diffs = np.diff(positions)
            if not np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-3) or diffs[0] <= 0:
                raise ValueError("DICOM slice positions are not a regular stack")
            dz = float(diffs[0])
            displacement = np.array(datasets[-1].ImagePositionPatient)-np.array(datasets[0].ImagePositionPatient)
            if not np.allclose(displacement, normal*dz*(len(datasets)-1), atol=1e-2):
                raise ValueError("Gantry tilt requires explicit resampling")
        dy, dx = map(float, datasets[0].PixelSpacing)
        if any(not np.allclose(x.PixelSpacing, [dy, dx]) for x in datasets):
            raise ValueError("Inconsistent DICOM spacing")
        array = np.stack([x.pixel_array.astype(np.float32)*float(getattr(x,'RescaleSlope',1))
                          +float(getattr(x,'RescaleIntercept',0)) for x in datasets])
        affine = np.eye(4)
        affine[:3,0], affine[:3,1], affine[:3,2] = orient[:3]*dx, orient[3:]*dy, normal*dz
        affine[:3,3] = datasets[0].ImagePositionPatient
        affine = np.diag([-1., -1., 1., 1.]) @ affine  # LPS to RAS.
        # Patient/Study tags are deliberately discarded, never sent to the client.
        return Volume(array, affine, (dz, dy, dx), 'mm', 'DICOM')
    raise ValueError("Supported formats: NIfTI, PNG/JPEG, single-frame DICOM or DICOM-series ZIP")


def save_volume(path, array, affine, units='mm'):
    image = nib.Nifti1Image(np.asarray(array).transpose(2,1,0), affine)
    if units == 'mm':
        image.header.set_xyzt_units('mm')
    nib.save(image, path)


def phantom(shape=(48, 128, 128), seed=42):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(-1,1,n) for n in shape], indexing='ij')
    body = x*x/0.8**2+y*y/0.7**2 < 1
    organ = ((x+0.2)/0.32)**2+((y-0.04)/0.3)**2+(z/0.7)**2 < 1
    lesion = ((x+0.25)/0.09)**2+((y-0.01)/0.08)**2+((z-0.05)/0.18)**2 < 1
    image = np.full(shape, -1000, dtype=np.float32)
    image[body] = 30
    image[organ] = 125
    image[lesion] = 65
    image += rng.normal(0, 7, shape).astype(np.float32)
    affine = np.diag([1.2, 1.2, 2.5, 1.])
    return Volume(image, affine, (2.5, 1.2, 1.2), 'mm', 'Generated CT phantom'), organ.astype(np.uint8)
