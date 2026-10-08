import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import torch

from gpu_runtime.adapters import SegmentationModel
from gpu_runtime.engine import InferenceEngine
from gpu_runtime.io import save_volume,load_volume
from gpu_runtime.metrics import evaluate,failure_mining,regression
from gpu_runtime.store import Store


class IdentityAdapter(SegmentationModel):
    def load(self,device):
        return torch.nn.Identity().to(device)


class RuntimeTests(unittest.TestCase):
    def test_dicom_stack_sorting_scaling_ras_and_irregular_rejection(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d); series=generate_uid(); files=[]
            for index in [2,0,1]:
                meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID=CTImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
                path=directory/f'{index}.dcm'
                ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
                ds.is_little_endian=True;ds.is_implicit_VR=False;ds.SeriesInstanceUID=series
                ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[10,20,index*3]
                ds.PixelSpacing=[1.5,.8];ds.Rows=3;ds.Columns=4
                ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
                ds.BitsAllocated=ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=1
                ds.RescaleSlope=2;ds.RescaleIntercept=-100
                ds.PixelData=np.full((3,4),index,dtype=np.int16).tobytes();ds.save_as(path)
                files.append(path)
            archive=directory/'series.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for path in files:z.write(path,path.name)
            volume=load_volume(archive)
            np.testing.assert_array_equal(volume.array[:,0,0],[-100,-98,-96])
            np.testing.assert_allclose(volume.affine,np.array([[-.8,0,0,-10],[0,-1.5,0,-20],[0,0,3,0],[0,0,0,1]]))
            self.assertEqual(volume.spacing,(3.,1.5,.8))
            ds.ImagePositionPatient=[10,20,3.5];ds.save_as(directory/'1.dcm')
            with zipfile.ZipFile(archive,'w') as z:
                for path in files:z.write(path,path.name)
            with self.assertRaisesRegex(ValueError,'regular stack'):load_volume(archive)

    def test_sliding_window_batch_and_native_restoration(self):
        volume=np.full((2,9,25,21),2,dtype=np.float32)
        result=InferenceEngine(IdentityAdapter()).infer(volume,roi_size=(16,16,16),batch_size=2,
                                                       resample_spacing=(.8,.8,.8))
        self.assertEqual(result['mask'].shape,volume.shape)
        np.testing.assert_array_equal(result['mask'],np.ones_like(volume))
        self.assertGreater(result['timings']['total_ms'],0)

    def test_fp16_cpu_rejected_and_bad_adapter_shape(self):
        engine=InferenceEngine(IdentityAdapter())
        with self.assertRaises(ValueError):engine.infer(np.zeros((16,16,16)),precision='fp16')
        with self.assertRaises(ValueError):engine.infer(np.full((16,16,16),np.nan))

    def test_native_nifti_axis_and_affine_roundtrip(self):
        volume=np.arange(5*7*9,dtype=np.float32).reshape(5,7,9)
        affine=np.array([[0,-1.3,0,30],[.8,0,0,-10],[0,0,2.5,17],[0,0,0,1.]])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'volume.nii.gz';save_volume(path,volume,affine)
            result=load_volume(path)
        np.testing.assert_array_equal(result.array,volume)
        np.testing.assert_allclose(result.affine,affine)
        np.testing.assert_allclose(result.spacing,(2.5,1.3,.8))

    def test_anisotropic_surface_distance_and_empty_masks(self):
        a=np.zeros((9,9,9),np.uint8);b=a.copy();a[4,4,4]=1;b[5,4,4]=1
        self.assertAlmostEqual(evaluate(a,b,(3,1,1))['hd95'],3.)
        empty=np.zeros_like(a)
        self.assertIsNone(evaluate(a,empty)['hd95'])
        self.assertEqual(evaluate(empty,empty)['dice'],1.)

    def test_fp_fn_only_with_reference(self):
        pred=np.zeros((5,5,5),np.uint8);pred[1:3,1:3,1:3]=1
        uncertain=np.zeros_like(pred,dtype=float);uncertain[3:5,3:5,3:5]=1
        a=failure_mining(pred,uncertain)
        self.assertEqual({x['kind'] for x in a['regions']},{'low_confidence'})
        b=failure_mining(pred,uncertain,np.zeros_like(pred))
        self.assertIn('FP_vs_reference',{x['kind'] for x in b['regions']})

    def test_regression_detects_accuracy_and_performance_failure(self):
        a={'dice':.9,'p50_ms':10,'peak_allocated_mib':100}
        b={'dice':.8,'p50_ms':15,'peak_allocated_mib':130}
        self.assertFalse(regression(a,b)['passed'])

    def test_review_survives_reopen_and_records_error_pool(self):
        with tempfile.TemporaryDirectory() as d:
            db=Store(d);case,_=db.new_case({'source':'synthetic'});run,_=db.new_run(case['id'],{})
            db.review(run['id'],'reject',taxonomy='boundary_deviation')
            reopened=Store(d)
            self.assertEqual(reopened.run(run['id'])[0]['review'],'reject')
            self.assertEqual(reopened.hard_cases()[0]['taxonomy'],'boundary_deviation')


if __name__=='__main__':unittest.main()
