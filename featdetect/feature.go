package featdetect

// FeatType : Feature type enum
type FeatType int

const (
	// DoG : Difference of Gaussians feature type
	DoG FeatType = 0
	// Harris : Harris feature type
	Harris FeatType = 1
)

// Feature : Represents a detected feature in an image
type Feature struct {
	X        int
	Y        int
	Response float32
	Type     FeatType
}

// NewFeature : Creates new feature with the given specifications
func NewFeature(x, y int, response float32, featType FeatType) *Feature {
	feat := new(Feature)
	feat.X, feat.Y, feat.Response, feat.Type = x, y, response, featType
	return feat
}
