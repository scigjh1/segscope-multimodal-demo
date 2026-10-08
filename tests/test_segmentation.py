import base64,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import segment,private_model_mask,load_private_predictor

class SegmentationTests(unittest.TestCase):
 def tearDown(self):load_private_predictor.cache_clear()
 def test_fast_roundtrip_preserves_output_shape(self):
  image=np.zeros((80,100,3),np.uint8);image[20:60,25:75]=180
  raw=cv2.imencode('.png',image)[1].tobytes()
  result=segment({'image':'data:image/png;base64,'+base64.b64encode(raw).decode(),'mode':'fast'})
  decoded=cv2.imdecode(np.frombuffer(base64.b64decode(result['fusedMask'].split(',')[1]),np.uint8),cv2.IMREAD_GRAYSCALE)
  self.assertEqual(decoded.shape,(80,100));self.assertEqual(set(np.unique(decoded)),{0,255})
 def test_private_predictor_contract_and_rejection(self):
  image=np.zeros((12,16,3),np.uint8)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'predictor.py';path.write_text('import numpy as np\ndef predict(image,aux,roi):\n m=np.zeros(image.shape[:2],np.uint8);m[3:8,4:12]=1;return m\n')
   with patch.dict(os.environ,{'SEG_SCOPE_PREDICTOR_FILE':str(path)}):
    mask=private_model_mask(image,None,(0,0,16,12));self.assertEqual(int(mask.sum()),40*255)
   path.write_text('import numpy as np\ndef predict(image,aux,roi):return np.ones((2,2))\n');load_private_predictor.cache_clear()
   with patch.dict(os.environ,{'SEG_SCOPE_PREDICTOR_FILE':str(path)}):
    with self.assertRaises(ValueError):private_model_mask(image,None,(0,0,16,12))
 def test_unconfigured_private_predictor_is_explicit(self):
  with patch.dict(os.environ,{'SEG_SCOPE_PREDICTOR_FILE':''}):
   with self.assertRaises(ValueError):private_model_mask(np.zeros((10,10,3),np.uint8),None,(0,0,10,10))

if __name__=='__main__':unittest.main()
