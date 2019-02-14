package structs

import (
	"image"
	"image/color"
)

type Float32 float32

type RGB struct {
	R float32
	G float32
	B float32
}

type Photo struct {
	Width  int
	Height int
	Data   [][]RGB
}

type GrayPhoto struct {
	Width  int
	Height int
	Data   [][]Float32
}

// RGBA : Used to implement image.Color interface
func (rgb *RGB) RGBA() (r, g, b, a uint32) {
	return uint32(rgb.R * 0xffff),
		uint32(rgb.G * 0xffff),
		uint32(rgb.B * 0xffff), 0xffff
}

// RGBA : Used to implement image.Color interface
func (gray Float32) RGBA() (r, g, b, a uint32) {
	return uint32(gray * 0xffff),
		uint32(gray * 0xffff),
		uint32(gray * 0xffff), 0xffff
}

// ColorModel : Used to implement image.Image interface
func (photo *Photo) ColorModel() color.Model {
	return color.RGBAModel
}

// ColorModel : Used to implement image.Image interface
func (photo *GrayPhoto) ColorModel() color.Model {
	return color.GrayModel
}

// At : Used to implement image.Image interface
func (photo *Photo) At(x, y int) color.Color {
	return &photo.Data[y][x]
}

// At : Used to implement image.Image interface
func (photo *GrayPhoto) At(x, y int) color.Color {
	return photo.Data[y][x]
}

// Bounds : Used to implement image.Image interface
func (photo *Photo) Bounds() image.Rectangle {
	return image.Rect(0, 0, photo.Width, photo.Height)
}

// Bounds : Used to implement image.Image interface
func (photo *GrayPhoto) Bounds() image.Rectangle {
	return image.Rect(0, 0, photo.Width, photo.Height)
}

func (photo *GrayPhoto) SubtractPhoto(photo2 *GrayPhoto) {
	for y := 0; y < photo.Height; y++ {
		for x := 0; x < photo.Width; x++ {
			photo.Data[y][x] -= photo2.Data[y][x]
		}
	}
	return
}

func (photo *GrayPhoto) SubtractImage(img image.Image) {
	for y := 0; y < photo.Height; y++ {
		for x := 0; x < photo.Width; x++ {
			r, _, _, a := img.At(x, y).RGBA()
			photo.Data[y][x] -= Float32(r) / Float32(a)
		}
	}
	return
}
