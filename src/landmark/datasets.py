"""Public test data sets to use in testing or showcasing k-center problems."""
import numpy as np
import urllib.request

_SHAPE_DATA_URLS = {
	"aggregation": "https://cs.joensuu.fi/sipu/datasets/Aggregation.txt",
	"compound": "https://cs.joensuu.fi/sipu/datasets/Compound.txt",
	"pathbased": "https://cs.joensuu.fi/sipu/datasets/pathbased.txt",
	"spiral": "https://cs.joensuu.fi/sipu/datasets/spiral.txt",
	"d31": "https://cs.joensuu.fi/sipu/datasets/D31.txt",
	"r15": "https://cs.joensuu.fi/sipu/datasets/R15.txt",
	"jain": "https://cs.joensuu.fi/sipu/datasets/jain.txt",
	"flame": "https://cs.joensuu.fi/sipu/datasets/flame.txt",
}


def load_shape(shape: str) -> np.ndarray:
	"""Loads a shape dataset from the 'Clustering basic benchmark' by P. Fänti and S. Sieranoja."""
	assert (
		shape.lower() in _SHAPE_DATA_URLS.keys()
	), f"Invalid shape name {shape} given; must be one of {list(_SHAPE_DATA_URLS.keys())}"

	## Note: we don't use loadtxt here because that saves a directory in the local
	url = _SHAPE_DATA_URLS[shape.lower()]
	response = urllib.request.urlopen(url)
	data = response.read()  # a `bytes` object
	X = np.fromstring(data.decode("utf-8"), sep="\t")
	return np.array(X.reshape(len(X) // 3, 3)).astype(np.float32)
