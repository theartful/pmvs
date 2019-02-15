package image

func ConvolveX(photo *CHWImage, filter []float32) *CHWImage {
	height, width, channel := photo.Height, photo.Width, photo.Channel
	dataLength := channel * height * width

	result := NewImage(height, width, channel)
	filterLength := len(filter)
	halfLength := filterLength / 2

	for i := halfLength; i < dataLength-halfLength; i++ {
		var val float32
		for k := 0; k < filterLength; k++ {
			val += photo.Data[i+k-halfLength] * filter[k]
		}
		result.Data[i] = val
	}
	return result
}

func ConvolveY(photo *CHWImage, filter []float32) *CHWImage {
	height, width, channel := photo.Height, photo.Width, photo.Channel
	dataLength := channel * height * width
	result := NewImage(height, width, channel)
	filterLength := len(filter)
	halfLength := filterLength / 2

	for i := halfLength * width; i < dataLength-halfLength*width; i++ {
		var val float32
		convIndex := i - halfLength*width
		for k := 0; k < filterLength; k++ {
			val += photo.Data[convIndex] * filter[k]
			convIndex += width
		}
		result.Data[i] = val
	}

	return result
}
