package core

import (
	"math"
	"pmvs/featdetect"

	"gonum.org/v1/gonum/mat"
)

var (
	// used in getPatchVectors
	// defined here instead of redefining it each time
	bForPatchVecs = mat.NewDense(4, 2, []float64{
		1, 0,
		0, 1,
		0, 0,
		0, 0,
	})
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

func getPatchVectors(patch *Patch) (right, up *mat.VecDense) {
	proj := imgsManager.Photos[patch.RefImg].CameraMatrix()
	normal := patch.Normal
	center := patch.Center
	// there is probably a better way to do this
	a := mat.NewDense(4, 4, []float64{
		proj.At(0, 0), proj.At(0, 1), proj.At(0, 2), proj.At(0, 3),
		proj.At(1, 0), proj.At(1, 1), proj.At(1, 2), proj.At(1, 3),
		proj.At(2, 0), proj.At(2, 1), proj.At(2, 2), proj.At(2, 3),
		normal.AtVec(0), normal.AtVec(1), normal.AtVec(2), normal.AtVec(3),
	})

	sol := mat.NewDense(4, 2, nil)
	sol.Solve(a, bForPatchVecs)
	// copy results
	right = mat.NewVecDense(4, nil)
	up = mat.NewVecDense(4, nil)
	right.CloneVec(sol.ColView(0))
	up.CloneVec(sol.ColView(1))
	// scaling
	d := mat.Dot(center, proj.RowView(2))
	right.ScaleVec(d, right)
	up.ScaleVec(d, up)
	return
}

// triangulate : finds a point p such that normalized(proj1 * p) = (x1, y1, 1)
// and: ||normalized(proj2 * x) - (x2, y2, 1)|| is minimized
func triangulate(x1, y1, x2, y2, id1, id2 int) *mat.VecDense {
	proj1 := imgsManager.Photos[id1].CameraMatrix()
	proj2 := imgsManager.Photos[id2].CameraMatrix()
	funMat := imgsManager.FundamentalMatrix(id1, id2)
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
