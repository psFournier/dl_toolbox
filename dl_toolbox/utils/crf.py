from argparse import ArgumentParser
import rasterio
import dl_toolbox.inference as dl_inf
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--probas_path", type=str)
args = parser.parse_args()

probas = dl_inf.read_probas(args.probas_path)[4,...]
profile = rasterio.open(args.probas_path).profile

preds_basic = probas > 0.5
probas = np.vstack([1-probas, probas])
unary = -np.log(probas).reshape((2, -1)).astype(np.float32)
d = dcrf.DenseCRF2D(2000, 2000, 2)
d.setUnaryEnergy(unary)
d.addPairwiseGaussian(sxy=3, compat=3)
Q = d.inference(10)
MAP = np.argmax(Q, axis=0)
preds_crf = MAP.reshape((2000,2000))

#preds_crf = np.expand_dims(preds_crf, 0)
#dl_inf.write_array(
#    preds_crf,
#    (0,0,2000,2000),
#    '/home/pfournie/ai4geo/outputs/toulouse_tuile_9_crf.tif',
#    profile
#)

fig = plt.figure(figsize=(40,80))
ax1 = fig.add_subplot(121)
ax1.imshow(preds_basic)
ax2 = fig.add_subplot(122)
ax2.imshow(preds_crf)
plt.savefig('/home/pfournie/ai4geo/outputs/crf.jpg', dpi=600)
