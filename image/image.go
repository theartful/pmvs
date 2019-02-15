package image

// CHWImage : A custom struct that represents an image as a continguous array of data
type CHWImage struct {
	Width   int
	Height  int
	Channel int
	Data    []float32
}

// At : Returns the value at (y, x, c)
func (image *CHWImage) At(y, x, c int) float32 {
	return image.Data[c*image.Width*image.Height+y*image.Width+x]
}

// Set : Sets the value at (y, x, c) to be val
func (image *CHWImage) Set(y, x, c int, val float32) {
	image.Data[c*image.Width*image.Height+y*image.Width+x] = val
}

func (image *CHWImage) Subtract(image2 *CHWImage) *CHWImage {
	length := len(image.Data)
	for i := 0; i < length; i++ {
		image.Data[i] -= image2.Data[i]
	}
	return image
}

func (image *CHWImage) Add(image2 *CHWImage) *CHWImage {
	length := len(image.Data)
	for i := 0; i < length; i++ {
		image.Data[i] += image2.Data[i]
	}
	return image
}

func (image *CHWImage) Mul(image2 *CHWImage) *CHWImage {
	length := len(image.Data)
	for i := 0; i < length; i++ {
		image.Data[i] *= image2.Data[i]
	}
	return image
}

func Subtract(image, image2 *CHWImage) *CHWImage {
	image3 := NewImage(image.Height, image.Width, image.Channel)
	length := len(image.Data)
	for i := 0; i < length; i++ {
		image3.Data[i] = image.Data[i] - image2.Data[i]
	}
	return image3
}

func Add(image, image2 *CHWImage) *CHWImage {
	image3 := NewImage(image.Height, image.Width, image.Channel)
	length := len(image.Data)
	for i := 0; i < length; i++ {
		image3.Data[i] = image.Data[i] + image2.Data[i]
	}
	return image3
}

func Mul(image, image2 *CHWImage) *CHWImage {
	image3 := NewImage(image.Height, image.Width, image.Channel)
	length := len(image.Data)
	for i := 0; i < length; i++ {
		image3.Data[i] = image.Data[i] * image2.Data[i]
	}
	return image3
}

// NewImage : Creates new image
func NewImage(height, width, channel int) *CHWImage {
	image := new(CHWImage)
	image.Height, image.Width, image.Channel = height, width, channel
	size := channel * height * width
	image.Data = make([]float32, size, size)
	return image
}
