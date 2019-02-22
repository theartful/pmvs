package core

import (
	"math"
	"pmvs/featdetect"

	"gonum.org/v1/gonum/mat"
)

// getRelevantFeatures : Matches a feature with possible candidate features
// this is done using the epipolar consistency metric
func getRelevantFeatures(
	feat *featdetect.Feature,
	featImgID int, searchIDs []int,
) (relevantFeats []*featdetect.Feature, correspondingIds []int) {

	featCoord := mat.NewVecDense(3, []float64{
		float64(feat.X), float64(feat.Y), 1,
	})

	epiLine := mat.NewVecDense(3, nil)
	for _, id := range searchIDs {
		if id == featImgID {
			continue
		}
		photo := imgsManager.Photos[id]
		searchFeats := photo.Feats[int(feat.Type)]

		funMat := imgsManager.FundamentalMatrix(featImgID, id)
		epiLine.MulVec(funMat, featCoord)

		maxDist := featMaxDist * math.Sqrt(
			epiLine.AtVec(0)*epiLine.AtVec(0)+epiLine.AtVec(1)*epiLine.AtVec(1),
		)

		for _, feat2 := range searchFeats {
			epiDist := math.Abs(epiLine.AtVec(0)*float64(feat2.X) +
				epiLine.AtVec(1)*float64(feat2.Y) + epiLine.AtVec(2))
			if epiDist <= maxDist {
				relevantFeats = append(relevantFeats, feat2)
				correspondingIds = append(correspondingIds, id)
			}
		}
	}
	return
}

// getRelevantImages : Find images that look at the same parts
func getRelevantImages(id int) []int {
	img := imgsManager.Photos[id]
	opticalAxis1 := img.OpticalAxis()
	relevantImgs := make([]int, 0, 5)
	for i := 0; i < len(imgsManager.Photos); i++ {
		if i == id {
			continue
		}
		img2 := imgsManager.Photos[i]
		opticalAxis2 := img2.OpticalAxis()
		cosAngle := mat.Dot(opticalAxis1, opticalAxis2)
		if cosAngle > cosMaxAngle && cosAngle < cosMinAngle {
			relevantImgs = append(relevantImgs, i)
		}
	}
	return relevantImgs
}

func getPatchVectors(photo *Photo, center, normal *mat.VecDense) (right, up *mat.VecDense) {
	proj := photo.Cam.ProjMat
	// right and up vectors are the first and second column of the
	// pseudo inverse of the projection matrix
	pinv := photo.Cam.Pinv

	// subtract the normal component from right vector
	// now right lies on the plane defined by normal
	scale := pinv.At(0, 0)*normal.AtVec(0) +
		pinv.At(1, 0)*normal.AtVec(1) +
		pinv.At(2, 0)*normal.AtVec(2)
	right = mat.NewVecDense(4, []float64{
		pinv.At(0, 0) - scale*normal.AtVec(0),
		pinv.At(1, 0) - scale*normal.AtVec(1),
		pinv.At(2, 0) - scale*normal.AtVec(2),
		0,
	})

	// subtract the normal component from up vector
	// now up lies on the plane defined by normal
	scale = pinv.At(0, 1)*normal.AtVec(0) +
		pinv.At(1, 1)*normal.AtVec(1) +
		pinv.At(2, 1)*normal.AtVec(2)
	up = mat.NewVecDense(4, []float64{
		pinv.At(0, 1) - scale*normal.AtVec(0),
		pinv.At(1, 1) - scale*normal.AtVec(1),
		pinv.At(2, 1) - scale*normal.AtVec(2),
		0,
	})

	scale = mat.Dot(center, proj.RowView(2))
	right.ScaleVec(scale/mat.Dot(proj.RowView(0), right), right)
	up.ScaleVec(scale/mat.Dot(proj.RowView(1), up), up)
	return
}

// triangulate : finds a point p such that normalized(proj1 * p) = (x1, y1, 1)
// and: ||normalized(proj2 * x) - (x2, y2, 1)|| is minimized
func triangulate(x1, y1, x2, y2, id1, id2 int) *mat.VecDense {
	proj1 := imgsManager.Photos[id1].CameraMatrix()
	proj2 := imgsManager.Photos[id2].CameraMatrix()
	funMat := imgsManager.FundamentalMatrix(id1, id2)
	return _triangulate(x1, y1, x2, y2, proj1, proj2, funMat)
}

func _triangulate(x1, y1, x2, y2 int, proj1, proj2, funMat *mat.Dense) *mat.VecDense {
	x1float, y1float, x2float, y2float :=
		float64(x1), float64(y1), float64(x2), float64(y2)

	b := mat.NewVecDense(4, []float64{
		x1float, y1float, 1, 0,
	})

	line := mat.NewVecDense(3, nil)
	line.MulVec(funMat, b.SliceVec(0, 3))

	linePerp := mat.NewVecDense(3, []float64{
		-line.AtVec(1), line.AtVec(0),
		(line.AtVec(1)*x2float - line.AtVec(0)*y2float),
	})

	temp := mat.NewVecDense(4, nil)
	temp.MulVec(proj2.T(), linePerp)

	// there is probably a better way to do this
	a := mat.NewDense(4, 4, []float64{
		proj1.At(0, 0), proj1.At(0, 1), proj1.At(0, 2), proj1.At(0, 3),
		proj1.At(1, 0), proj1.At(1, 1), proj1.At(1, 2), proj1.At(1, 3),
		proj1.At(2, 0), proj1.At(2, 1), proj1.At(2, 2), proj1.At(2, 3),
		temp.AtVec(0), temp.AtVec(1), temp.AtVec(2), temp.AtVec(3),
	})

	sol := mat.NewVecDense(4, nil)
	sol.SolveVec(a, b)
	return sol
}

func projectGrid(photo *Photo, center, right, up *mat.VecDense,
	gridSize int, result []float32) []float32 {

	// project patch center on the photo
	projMat := photo.CameraMatrix()
	projCenter := mat.NewVecDense(3, nil)
	projCenter.MulVec(projMat, center)

	// project patch vectors on photo
	projRight, projUp := mat.NewVecDense(3, nil), mat.NewVecDense(3, nil)
	projRight.MulVec(projMat, right)
	projUp.MulVec(projMat, up)
	scale := 1.0 / projCenter.AtVec(2)
	projRight.ScaleVec(scale, projRight)
	projUp.ScaleVec(scale, projUp)
	projCenter.ScaleVec(scale, projCenter)

	step := float64(gridSize-1) / 2.0
	diag := mat.NewVecDense(3, nil)
	diag.AddVec(projRight, projUp)
	diag.ScaleVec(step, diag)
	topLeft := diag
	topLeft.SubVec(projCenter, diag)

	resIndex := 0
	for i := 0; i < gridSize; i++ {
		for j := 0; j < gridSize; j++ {
			ifloat, jfloat := float64(i), float64(j)
			x := topLeft.AtVec(0) + ifloat*projUp.AtVec(0) + jfloat*projRight.AtVec(0)
			y := topLeft.AtVec(1) + ifloat*projUp.AtVec(1) + jfloat*projRight.AtVec(1)
			result[resIndex], result[resIndex+1], result[resIndex+2] = photo.At(y, x)
			resIndex += 3
		}
	}
	return result
}

func patchNCCScore(photo *Photo, patch *Patch, right, up *mat.VecDense) float64 {
	refPhoto := imgsManager.Photos[patch.RefPhoto]
	cell1 := make([]float32, patchGridSize*patchGridSize*3)
	cell2 := make([]float32, patchGridSize*patchGridSize*3)

	cell1 = projectGrid(refPhoto, patch.Center, right, up, patchGridSize, cell1)
	cell2 = projectGrid(photo, patch.Center, right, up, patchGridSize, cell2)
	return ncc(cell1, cell2)
}

func ncc(cell1 []float32, cell2 []float32) float64 {
	var mean1, mean2 float32
	length := len(cell1)
	for i := 0; i < length; i++ {
		mean1 += cell1[i]
		mean2 += cell2[i]
	}
	mean1 /= float32(length)
	mean2 /= float32(length)

	var std1, std2, product float32
	for i := 0; i < length; i++ {
		diff1 := cell1[i] - mean1
		diff2 := cell2[i] - mean2
		product += diff1 * diff2
		std1 += diff1 * diff1
		std2 += diff2 * diff2
	}
	stds := std1 * std2
	if stds == 0 {
		return 0
	}
	return float64(product) / math.Sqrt(float64(stds))
}

func visualHullCheck(point *mat.VecDense) bool {
	projectedPoint := mat.NewVecDense(3, nil)
	for _, photo := range imgsManager.Photos {
		if photo.Mask == nil {
			continue
		}
		projectedPoint.MulVec(photo.CameraMatrix(), point)
		scale := projectedPoint.AtVec(2)
		if scale == 0 {
			continue
		}
		if photo.IsMasked(projectedPoint.AtVec(1)/scale,
			projectedPoint.AtVec(0)/scale) {
			return false
		}
	}
	return true
}

func constraintPhotos(patch *Patch, minNCC float64, searchIDs []int) []int {
	refPhoto := imgsManager.Photos[patch.RefPhoto]
	right, up := getPatchVectors(refPhoto, patch.Center, patch.Normal)
	result := make([]int, 0, 5)
	depthVector := mat.NewVecDense(4, nil)

	for _, photoID := range searchIDs {
		photo := imgsManager.Photos[photoID]
		depthVector.SubVec(photo.OpticalCenter(), patch.Center)
		if mat.Dot(depthVector, patch.Normal) <= 0 {
			continue
		}
		nccScore := patchNCCScore(photo, patch, right, up)
		if nccScore >= minNCC {
			result = append(result, photoID)
		}
	}
	return result
}

func registerPatch(patch *Patch) {
	photoCoord := mat.NewVecDense(3, nil)
	for _, photoID := range patch.TPhotos {
		photo := imgsManager.Photos[photoID]
		photoCoord.MulVec(photo.CameraMatrix(), patch.Center)
		x := int(photoCoord.AtVec(0)/photoCoord.AtVec(2)) / cellSize
		y := int(photoCoord.AtVec(1)/photoCoord.AtVec(2)) / cellSize
		cell := photo.Cells[y][x]
		cell.Patches = append(cell.Patches, patch)
	}
	imgsManager.Patches = append(imgsManager.Patches, patch)
}

func getCell(photoID, y, x int) *Cell {
	cellY := y / cellSize
	cellX := x / cellSize
	return imgsManager.Photos[photoID].Cells[cellY][cellX]
}
