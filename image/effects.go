package image

import "math"

// Grayscale : Transforms n-channel image into grayscale
// (not really a gray scale)
func Grayscale(image *CHWImage) *CHWImage {
	grayImage := NewImage(image.Height, image.Width, 1)
	imageSize := image.Height * image.Width
	for i := 0; i < imageSize; i++ {
		grayImage.Data[i] = image.Data[i]*image.Data[i] +
			image.Data[i+imageSize]*image.Data[i+imageSize] +
			image.Data[i+2*imageSize]*image.Data[i+2*imageSize]
	}
	return grayImage
}

// GaussianFilter : Apply gaussian filter and return new image
func GaussianFilter(photo *CHWImage, sigma float64) *CHWImage {
	filterSize := int(math.Ceil(2*sigma) + 1)
	halfSize := filterSize / 2
	filter := make([]float32, filterSize, filterSize)
	var sum float32
	for i := 0; i < filterSize; i++ {
		filter[i] = float32(math.Exp(-float64((i-halfSize)*(i-halfSize)) / (2.0 * sigma * sigma)))
		sum += filter[i]
	}
	for i := 0; i < filterSize; i++ {
		filter[i] /= sum
	}
	return ConvolveX(ConvolveY(photo, filter), filter)
}
