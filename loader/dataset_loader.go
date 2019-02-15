package loader

import (
	"bufio"
	"errors"
	"fmt"
	goimage "image"
	"os"
	"pmvs/image"
	"strconv"
	"strings"

	// for decoding images
	_ "image/jpeg"
	_ "image/png"

	// for decoding ppm, pgm images
	_ "github.com/jbuchbinder/gopnm"
)

var (
	supportedExtensions = map[string]bool{
		"jpeg": true,
		"jpg":  true,
		"png":  true,
		"ppm":  true,
		"pgm":  true,
	}

	errNotSupportedExt = errors.New("Error! Extension is not supported")
	errEmptyFile       = errors.New("Error! Empty txt file")
)

// LoadDataset : Loads a dataset consisting of images, projection matrices, and
// possibly masks. The images must be in "path/images", masks in "path/silhouettes",
// and matrices in "path/calib". each matrix in 'mats' is in row-major
func LoadDataset(
	path string,
	ext string,
	mask bool,
	maskExt string,
) (images, silhouettes []*image.CHWImage, mats [][]float64, err error) {

	if !supportedExtensions[ext] {
		err = errNotSupportedExt
		return
	}
	if mask && !supportedExtensions[maskExt] {
		err = errNotSupportedExt
		return
	}

	if !strings.HasSuffix(path, "/") {
		path += "/"
	}

	images = make([]*image.CHWImage, 0, 20)
	silhouettes = make([]*image.CHWImage, 0, 20)

	for i := 0; ; i++ {
		imageData, _, errLoad := loadImage(
			fmt.Sprintf("%simages/%04d.%s", path, i, ext))
		if errLoad != nil {
			break
		}
		images = append(images, To3HWImage(imageData))

		imageData, _, err = loadImage(
			fmt.Sprintf("%ssilhouettes/%04d.%s", path, i, maskExt))
		if err != nil {
			break
		}
		silhouettes = append(silhouettes, To1HWImage(imageData))

		var mat []float64
		mat, err = loadProjMatrix(fmt.Sprintf("%scalib/%04d.%s", path, i, "txt"))
		if err != nil {
			break
		}
		mats = append(mats, mat)
	}
	return
}

func loadImage(path string) (imageData goimage.Image, imageType string, err error) {
	var imageFile *os.File
	imageFile, err = os.Open(path)
	if err != nil {
		return
	}
	defer imageFile.Close()
	imageData, imageType, err = goimage.Decode(imageFile)
	return
}

func loadProjMatrix(path string) (data []float64, err error) {
	reader, err := os.Open(path)
	if err != nil {
		return
	}
	defer reader.Close()
	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanWords)
	// read contour line
	if !scanner.Scan() {
		err = errEmptyFile
		return
	}
	for i := 0; i < 12; i++ {
		scanner.Scan()
		var num float64
		num, err = strconv.ParseFloat(scanner.Text(), 64)
		if err != nil {
			return
		}
		data = append(data, num)
	}
	return
}

func To3HWImage(img goimage.Image) *image.CHWImage {
	height, width := img.Bounds().Dy(), img.Bounds().Dx()
	chwImage := image.NewImage(height, width, 3)
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, a := img.At(x, y).RGBA()
			chwImage.Set(y, x, 0, float32(r)/float32(a))
			chwImage.Set(y, x, 1, float32(g)/float32(a))
			chwImage.Set(y, x, 2, float32(b)/float32(a))
		}
	}
	return chwImage
}

func To1HWImage(img goimage.Image) *image.CHWImage {
	height, width := img.Bounds().Dy(), img.Bounds().Dx()
	chwImage := image.NewImage(height, width, 1)
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, _, _, a := img.At(x, y).RGBA()
			chwImage.Set(y, x, 0, float32(r)/float32(a))
		}
	}
	return chwImage
}
