package core

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

var (
	photo       *Photo
	depthVec    *mat.VecDense
	optimPhotos []int
)

func encode(center, normal *mat.VecDense,
	photo *Photo, opticalCenter *mat.VecDense) (depth, theta, phi float64, depthVector *mat.VecDense) {

	depthVector = mat.NewVecDense(4, nil)
	depthVector.SubVec(center, opticalCenter)
	depth = math.Sqrt(mat.Dot(depthVector, depthVector))
	theta = math.Acos(normal.AtVec(2))
	phi = math.Atan2(normal.AtVec(1), normal.AtVec(0))
	depthVector.ScaleVec(1/depth, depthVector)
	return
}

func normalize(depth float64, unitDepthVector *mat.VecDense, photosIDs []int) (float64, *mat.VecDense) {
	depthVectorProj := mat.NewVecDense(3, nil)
	var sum float64
	for _, photoID := range photosIDs {
		photo := imgsManager.Photos[photoID]
		depthVectorProj.MulVec(photo.CameraMatrix(), unitDepthVector)
		depthVectorProj.ScaleVec(1/depthVectorProj.AtVec(2), depthVectorProj)
		sum += math.Sqrt(depthVectorProj.AtVec(0)*depthVectorProj.AtVec(0) +
			depthVectorProj.AtVec(1)*depthVectorProj.AtVec(1))
	}
	sum /= float64(len(photosIDs))
	unitDepthVector.ScaleVec(1/sum, unitDepthVector)
	depth *= sum
	return depth, unitDepthVector
}

func decode(photo *Photo, unitDepthVec *mat.VecDense,
	depth, theta, phi float64) (center, normal *mat.VecDense) {

	opticalCenter := photo.OpticalCenter()
	depthVector := mat.NewVecDense(4, nil)
	depthVector.ScaleVec(depth, unitDepthVec)
	center, normal = mat.NewVecDense(4, nil), mat.NewVecDense(4, nil)
	center.AddVec(opticalCenter, depthVector)
	normal.SetVec(0, math.Sin(theta)*math.Cos(phi))
	normal.SetVec(1, math.Sin(theta)*math.Sin(phi))
	normal.SetVec(2, math.Cos(theta))
	return center, normal
}

func nccObjective(center, right, up *mat.VecDense, refPhoto *Photo, targetPhotos []int) float64 {
	cell1 := make([]float32, patchGridSize*patchGridSize*3)
	cell2 := make([]float32, patchGridSize*patchGridSize*3)

	cell1 = projectGrid(refPhoto, center, right, up, patchGridSize, cell1)

	var totNcc float64
	for _, id := range targetPhotos {
		photo := imgsManager.Photos[id]
		cell2 = projectGrid(photo, center, right, up, patchGridSize, cell2)
		totNcc += ncc(cell1, cell2)
	}
	return totNcc / float64(len(targetPhotos))
}

func objectiveWrapper(x []float64) float64 {
	depth, theta, phi := x[0], x[1], x[2]
	center, normal := decode(photo, depthVec, depth, theta, phi)
	if !visualHullCheck(center) {
		return 1.0
	}
	if mat.Dot(depthVec, photo.OpticalAxis()) < 0 {
		return 1.0
	}
	right, up := getPatchVectors(photo, center, normal)
	return -nccObjective(center, right, up, photo, optimPhotos)
}

func optimizePatch(patch *Patch) {
	refPhoto := imgsManager.Photos[patch.RefPhoto]
	opticalCenter := refPhoto.OpticalCenter()

	depth, theta, phi, unitDepthVec :=
		encode(patch.Center, patch.Normal, refPhoto, opticalCenter)
	depth, unitDepthVec = normalize(depth, unitDepthVec, patch.TPhotos)
	targetPhotos := patch.TPhotos
	photo, depthVec, optimPhotos = refPhoto, unitDepthVec, targetPhotos

	problem := optimize.Problem{
		Func: objectiveWrapper,
		Grad: nil,
		Hess: nil,
	}
	method := optimize.NelderMead{
		SimplexSize: 1,
	}
	converger := optimize.FunctionConverge{
		Absolute:   0.0005,
		Iterations: 10,
	}
	settings := optimize.Settings{
		MajorIterations: 1000,
		Converger:       &converger,
	}

	result, _ := optimize.Minimize(problem, []float64{depth, theta, phi}, &settings, &method)
	center, normal := decode(refPhoto, unitDepthVec, result.Location.X[0],
		result.Location.X[1], result.Location.X[2])
	patch.Center, patch.Normal = center, normal
}
