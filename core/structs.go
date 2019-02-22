package core

import (
	"pmvs/featdetect"
	"pmvs/image"

	"gonum.org/v1/gonum/mat"
)

var (
	eye3 = mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
)

// ImagesManager : Container of input and output data
type ImagesManager struct {
	Photos   []*Photo
	FundMats [][]*mat.Dense
	Patches  []*Patch
}

// Photo : An image with its camera
type Photo struct {
	Img   *image.CHWImage
	Mask  *image.CHWImage
	Cam   *Camera
	Cells [][]*Cell
	Feats [][]*featdetect.Feature
	ID    int
}

// Cell : Photos are divided into cells that contain patches
type Cell struct {
	Patches []*Patch
}

// Camera : Relevant camera information
type Camera struct {
	ProjMat       *mat.Dense
	OpticalAxis   *mat.VecDense
	OpticalCenter *mat.VecDense
	Pinv          *mat.Dense
}

// Patch : A rectangle in 3D
type Patch struct {
	Normal   *mat.VecDense
	Center   *mat.VecDense
	RefPhoto int
	TPhotos  []int
}

// NewImagesManager : Creates new ImagesManager
// Sets the global variable "imgsManager" with the newely created manager
func NewImagesManager(imgs, masks []*image.CHWImage, projMats [][]float64) *ImagesManager {
	length := len(imgs)
	if length != len(projMats) {
		panic("Number of images and projection matrices aren't equal")
	}
	photos := make([]*Photo, length, length)
	fundMats := make([][]*mat.Dense, length, length)
	for i := 0; i < length; i++ {
		fundMats[i] = make([]*mat.Dense, length, length)
		photos[i] = newPhoto(imgs[i], masks[i], projMats[i], i)
	}
	imgsManager = new(ImagesManager)
	imgsManager.Photos, imgsManager.FundMats = photos, fundMats
	return imgsManager
}

func newPhoto(img, mask *image.CHWImage, projMat []float64, id int) *Photo {
	photo := new(Photo)
	photo.Img, photo.Mask = img, mask
	photo.ID = id
	photo.Cam = newCamera(projMat)
	cellsWidth := (photo.Img.Width + cellSize - 1) / cellSize
	cellsHeight := (photo.Img.Height + cellSize - 1) / cellSize
	photo.Cells = make([][]*Cell, cellsHeight, cellsHeight)
	for i := 0; i < cellsHeight; i++ {
		photo.Cells[i] = make([]*Cell, cellsWidth, cellsWidth)
		for j := 0; j < cellsWidth; j++ {
			photo.Cells[i][j] = new(Cell)
		}
	}

	photo.Cam.Pinv = mat.NewDense(4, 3, nil)
	photo.Cam.Pinv.Solve(photo.Cam.ProjMat, eye3)
	return photo
}

func newCamera(projMatData []float64) *Camera {
	if len(projMatData) != 3*4 {
		panic("Projection matrix should be of size 3x4")
	}
	projMat := mat.NewDense(3, 4, projMatData)

	// compute optical center
	mat3x3 := projMat.Slice(0, 3, 0, 3)
	opticalCenter := mat.NewVecDense(3, nil)
	opticalCenter.SolveVec(mat3x3, projMat.ColView(3))
	opticalCenter = mat.NewVecDense(4, []float64{
		-opticalCenter.AtVec(0), -opticalCenter.AtVec(1),
		-opticalCenter.AtVec(2), 1,
	})

	opticalAxis := mat.NewVecDense(4, []float64{
		projMat.At(2, 0), projMat.At(2, 1), projMat.At(2, 2), 0,
	})

	camera := new(Camera)
	camera.ProjMat = projMat
	camera.OpticalAxis = opticalAxis
	camera.OpticalCenter = opticalCenter
	return camera
}

// FundamentalMatrix : Return the fundamental matrix relating two images
func (imgsManager *ImagesManager) FundamentalMatrix(id1, id2 int) *mat.Dense {
	if imgsManager.FundMats[id1][id2] != nil {
		return imgsManager.FundMats[id1][id2]
	}
	photo1 := imgsManager.Photos[id1]
	photo2 := imgsManager.Photos[id2]
	proj2 := photo2.Cam.ProjMat
	opticalCenter1 := photo1.Cam.OpticalCenter

	ppinv := mat.NewDense(3, 3, nil)
	ppinv.Mul(proj2, photo1.Cam.Pinv)

	epipole := mat.NewVecDense(3, nil)
	epipole.MulVec(proj2, opticalCenter1)

	funMat := mat.NewDense(3, 3, nil)
	funMat.Mul(skewForm(epipole), ppinv)
	imgsManager.FundMats[id1][id2] = funMat
	imgsManager.FundMats[id2][id1] = mat.DenseCopyOf(funMat.T())
	return funMat
}

// CameraMatrix : Return the camera matrix
func (photo *Photo) CameraMatrix() *mat.Dense {
	return photo.Cam.ProjMat
}

// OpticalCenter : Return the optical center
func (photo *Photo) OpticalCenter() *mat.VecDense {
	return photo.Cam.OpticalCenter
}

// OpticalAxis : Return the optical axis
func (photo *Photo) OpticalAxis() *mat.VecDense {
	return photo.Cam.OpticalAxis
}

// At : Return the RGB color at (y, x)
func (photo *Photo) At(y, x float64) (r, g, b float32) {
	xint := int(x + 0.5)
	yint := int(y + 0.5)
	if xint < 0 || yint < 0 || xint >= photo.Img.Width ||
		yint >= photo.Img.Height {
		return 0, 0, 0
	}
	return photo.Img.At(yint, xint, 0), photo.Img.At(yint, xint, 1),
		photo.Img.At(yint, xint, 2)
}

// IsMasked : Return whether (y, x) is masked or not
func (photo *Photo) IsMasked(y, x float64) bool {
	xint := int(x + 0.5)
	yint := int(y + 0.5)
	// in this case the point lies outside the image bounadries
	// can't say whether it's masked
	if xint < 0 || yint < 0 || xint >= photo.Img.Width ||
		yint >= photo.Img.Height {
		return false
	}
	return photo.Mask.At(yint, xint, 0) == 0
}

// skewForm : skewForm(v).dot(u) = v cross u
func skewForm(vec *mat.VecDense) *mat.Dense {
	return mat.NewDense(3, 3, []float64{
		0, -vec.AtVec(2), vec.AtVec(1),
		vec.AtVec(2), 0, -vec.AtVec(0),
		-vec.AtVec(1), vec.AtVec(0), 0,
	})
}
