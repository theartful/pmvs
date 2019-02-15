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
	margin := GaussianMargin(sigma)
	filterSize := 2*margin + 1
	filter := make([]float32, filterSize, filterSize)
	var sum float32
	for i := 0; i < filterSize; i++ {
		filter[i] = float32(math.Exp(-float64((i-margin)*(i-margin)) /
			(2.0 * sigma * sigma)))
		sum += filter[i]
	}
	for i := 0; i < filterSize; i++ {
		filter[i] /= sum
	}
	return ConvolveX(ConvolveY(photo, filter), filter)
}

// GaussianMargin : Return the margin needed for gaussian filter of sigma std
func GaussianMargin(sigma float64) int {
	return int(math.Ceil(2 * sigma))
}

// HarrisCorner : Apply harris corner detector
func HarrisCorner(photo *CHWImage, sigma, k float64) *CHWImage {
	dFilter := []float32{-0.5, 0, 0.5}
	imgDx := ConvolveX(photo, dFilter)
	imgDy := ConvolveY(photo, dFilter)
	imgDxDy := GaussianFilter(Mul(imgDx, imgDy), sigma)
	imgDx2 := GaussianFilter(imgDx.Mul(imgDx), sigma)
	imgDy2 := GaussianFilter(imgDy.Mul(imgDy), sigma)

	arrLength := len(photo.Data)
	k32 := float32(k)
	harrisResponse := NewImage(photo.Height, photo.Width, photo.Channel)
	for i := 0; i < arrLength; i++ {
		det := imgDx2.Data[i]*imgDy2.Data[i] - imgDxDy.Data[i]*imgDxDy.Data[i]
		trace := imgDx2.Data[i] + imgDy2.Data[i]
		harrisResponse.Data[i] = det - k32*trace
	}

	harrisResponse2 := NewImage(photo.Height, photo.Width, photo.Channel)
	margin := GaussianMargin(sigma)
	for y := margin; y < harrisResponse.Height-margin; y++ {
		for x := margin; x < harrisResponse.Width-margin; x++ {
			val := harrisResponse.At(y, x, 0)
			if val < harrisResponse.At(y+1, x, 0) || val < harrisResponse.At(y-1, x, 0) ||
				val < harrisResponse.At(y+1, x+1, 0) || val < harrisResponse.At(y, x+1, 0) ||
				val < harrisResponse.At(y-1, x+1, 0) || val < harrisResponse.At(y+1, x-1, 0) ||
				val < harrisResponse.At(y, x-1, 0) || val < harrisResponse.At(y-1, x-1, 0) {
				harrisResponse2.Set(y, x, 0, 0)
			} else {
				harrisResponse2.Set(y, x, 0, val)
			}
		}
	}
	return harrisResponse2
}
