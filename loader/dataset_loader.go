package loader

import (
	"bufio"
	"errors"
	"fmt"
	"image"
	"os"
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
) (images, silhouettes []image.Image, mats [][]float64, err error) {

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

	images = make([]image.Image, 0, 20)
	silhouettes = make([]image.Image, 0, 20)

	for i := 0; ; i++ {
		imageData, _, errLoad := loadImage(
			fmt.Sprintf("%simages/%04d.%s", path, i, ext))
		if errLoad != nil {
			break
		}
		images = append(images, imageData)

		imageData, _, err = loadImage(
			fmt.Sprintf("%ssilhouettes/%04d.%s", path, i, maskExt))
		if err != nil {
			break
		}
		silhouettes = append(silhouettes, imageData)

		var mat []float64
		mat, err = loadProjMatrix(fmt.Sprintf("%scalib/%04d.%s", path, i, "txt"))
		if err != nil {
			break
		}
		mats = append(mats, mat)
	}
	return
}

func loadImage(path string) (imageData image.Image, imageType string, err error) {
	var imageFile *os.File
	imageFile, err = os.Open(path)
	if err != nil {
		return
	}
	defer imageFile.Close()
	imageData, imageType, err = image.Decode(imageFile)
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
