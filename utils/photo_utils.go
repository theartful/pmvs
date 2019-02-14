package utils

import (
	"image"
	"pmvs/structs"
)

// PhotoFromImage : creates a Photo from Image interface
func PhotoFromImage(img image.Image) *structs.Photo {
	var photo = new(structs.Photo)
	bounds := img.Bounds()
	photo.Width = bounds.Dx()
	photo.Height = bounds.Dy()

	photo.Data = make([][]structs.RGB, photo.Height, photo.Height)
	for y := 0; y < photo.Height; y++ {
		photo.Data[y] = make([]structs.RGB, photo.Width, photo.Width)
		for x := 0; x < photo.Width; x++ {
			r, g, b, a := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			photo.Data[y][x] = structs.RGB{R: float32(r) / float32(a),
				G: float32(g) / float32(a), B: float32(b) / float32(a)}
		}
	}
	return photo
}

// GrayPhotoFromImage : creates a GrayPhoto from Image interface using squared norm
// the purpose of this function is not to create a gray scale image for viewing
// but to get the response from a multichannel DoG operator
func GrayPhotoFromImage(img image.Image) *structs.GrayPhoto {
	var photo = new(structs.GrayPhoto)
	bounds := img.Bounds()
	photo.Width = bounds.Dx()
	photo.Height = bounds.Dy()

	photo.Data = make([][]structs.Float32, photo.Height, photo.Height)
	for y := 0; y < photo.Height; y++ {
		photo.Data[y] = make([]structs.Float32, photo.Width, photo.Width)
		for x := 0; x < photo.Width; x++ {
			r, g, b, a := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			photo.Data[y][x] = structs.Float32(r*r+g*g+b*b) / structs.Float32(a*a)
		}
	}
	return photo
}
