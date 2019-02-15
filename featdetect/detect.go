package featdetect

import "pmvs/image"

const (
	gridSize        int = 32
	featPerGridCell int = 4
)

// DetectFeatures : Detects DoG and Harris features from the image
func DetectFeatures(img, mask *image.CHWImage) []*Feature {
	features := detectDogFeatures(img, mask)
	features = append(features, detectHarrisFeatures(img, mask)...)
	return features
}
