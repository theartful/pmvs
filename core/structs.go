package core

import (
	"pmvs/featdetect"
	"pmvs/image"

	"gonum.org/v1/gonum/mat"
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
	Cam   *Camera
	Feats [][]*featdetect.Feature
	ID    int
}

// Camera : Relevant camera information
type Camera struct {
	ProjMat       *mat.Dense
	OpticalAxis   *mat.VecDense
	OpticalCenter *mat.VecDense
	pinv          *mat.Dense
}

// Patch : A rectangle in 3D
type Patch struct {
	Normal *mat.VecDense
	Center *mat.VecDense
	RefImg int
}

// NewImagesManager : Creates new ImagesManager
// Sets the global variable "imgsManager" with the newely created manager
func NewImagesManager(imgs []*image.CHWImage, projMats [][]float64) *ImagesManager {
	length := len(imgs)
	if length != len(projMats) {
		panic("Number of images and projection matrices aren't equal")
	}
	photos := make([]*Photo, length, length)
	fundMats := make([][]*mat.Dense, length, length)
	for i := 0; i < length; i++ {
		fundMats[i] = make([]*mat.Dense, length, length)
		photos[i] = newPhoto(imgs[i], projMats[i], i)
	}

	imgsManager = new(ImagesManager)
	imgsManager.Photos, imgsManager.FundMats = photos, fundMats
	return imgsManager
}

func newPhoto(img *image.CHWImage, projMat []float64, id int) *Photo {
	photo := new(Photo)
	photo.Img = img
	photo.Cam = newCamera(projMat)
	photo.ID = id
	return photo
}

func newCamera(projMatData []float64) *Camera {
	if len(projMatData) != 3*4 {
		panic("Projection matrix should be of size 3x4")
	}
	projMat := mat.NewDense(3, 4, projMatData)

	// compute optical center
	// probably not the best way
	mat3x3data := make([]float64, 9, 9)
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			mat3x3data[r*3+c] = projMat.At(r, c)
		}
	}
	mat3x3 := mat.NewDense(3, 3, mat3x3data)
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
	proj1 := photo1.Cam.ProjMat
	proj2 := photo2.Cam.ProjMat
	opticalCenter1 := photo1.Cam.OpticalCenter

	if photo1.Cam.pinv == nil {
		eye3 := mat.NewDense(3, 3, []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		})
		photo1.Cam.pinv = mat.NewDense(4, 3, nil)
		photo1.Cam.pinv.Solve(proj1, eye3)
	}
	ppinv := mat.NewDense(3, 3, nil)
	ppinv.Mul(proj2, photo1.Cam.pinv)

	epipole := mat.NewVecDense(3, nil)
	epipole.MulVec(proj2, opticalCenter1)

	funMat := mat.NewDense(3, 3, nil)
	funMat.Mul(skewForm(epipole), ppinv)
	imgsManager.FundMats[id1][id2] = funMat
	imgsManager.FundMats[id2][id1] = mat.DenseCopyOf(funMat.T())
	return funMat
}

// CameraMatrix : Return a copy of the camera matrix
func (photo *Photo) CameraMatrix() *mat.Dense {
	c := mat.NewDense(3, 4, nil)
	c.Clone(photo.Cam.ProjMat)
	return c
}

// OpticalCenter : Return a copy of the optical center
func (photo *Photo) OpticalCenter() *mat.VecDense {
	c := mat.NewVecDense(4, nil)
	c.CloneVec(photo.Cam.OpticalCenter)
	return c
}

// OpticalAxis : Return a copy of the optical axis
func (photo *Photo) OpticalAxis() *mat.VecDense {
	c := mat.NewVecDense(4, nil)
	c.CloneVec(photo.Cam.OpticalAxis)
	return c
}

func skewForm(vec *mat.VecDense) *mat.Dense {
	return mat.NewDense(3, 3, []float64{
		0, -vec.AtVec(2), vec.AtVec(1),
		vec.AtVec(2), 0, -vec.AtVec(0),
		-vec.AtVec(1), vec.AtVec(0), 0,
	})
}
