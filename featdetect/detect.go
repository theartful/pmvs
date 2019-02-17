package featdetect

import "pmvs/image"

const (
	gridSize        int = 32
	featPerGridCell int = 4
)

// DetectFeatures : Detects DoG and Harris features from the image
func DetectFeatures(img, mask *image.CHWImage) [][]*Feature {
	features := make([][]*Feature, 2, 2)
	features[0] = detectDogFeatures(img, mask)
	features[1] = detectHarrisFeatures(img, mask)
	return features
}
