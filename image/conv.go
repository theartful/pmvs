package image

// ConvolveX : Convolves an image with a 1d filter along the x-axis
func ConvolveX(photo *CHWImage, filter []float32) *CHWImage {
	height, width, channel := photo.Height, photo.Width, photo.Channel
	dataLength := channel * height * width

	result := NewImage(height, width, channel)
	filterLength := len(filter)
	margin := filterLength / 2

	for i := margin; i < dataLength-margin; i++ {
		var val float32
		convIndex := i - margin
		for k := 0; k < filterLength; k++ {
			val += photo.Data[convIndex] * filter[k]
			convIndex++
		}
		result.Data[i] = val
	}
	return result
}

// ConvolveY : Convolves an image with a 1d filter along the x-axis
func ConvolveY(photo *CHWImage, filter []float32) *CHWImage {
	height, width, channel := photo.Height, photo.Width, photo.Channel
	dataLength := channel * height * width
	result := NewImage(height, width, channel)
	filterLength := len(filter)
	margin := filterLength / 2

	for i := margin * width; i < dataLength-margin*width; i++ {
		var val float32
		convIndex := i - margin*width
		for k := 0; k < filterLength; k++ {
			val += photo.Data[convIndex] * filter[k]
			convIndex += width
		}
		result.Data[i] = val
	}

	return result
}
