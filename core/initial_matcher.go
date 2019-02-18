package core

import (
	"fmt"
	"math"
	"pmvs/featdetect"
	"sort"

	"gonum.org/v1/gonum/mat"
)

func StartMatching() {
	fmt.Println("Initial Matching...")

	for id, photo := range imgsManager.Photos {
		num := 0
		relevantImgs := getRelevantImages(id)
		for _, featPool := range photo.Feats {
			for _, feat := range featPool {
				num += constructPatch(id, relevantImgs, feat)
			}
		}
		fmt.Println("done img", id, " patches ", num)
	}
}

func constructPatch(photoID int, relevantImgs []int, feat *featdetect.Feature) int {
	type FeatSort struct {
		feature  *featdetect.Feature
		photoID  int
		relDepth float64
		pos3d    *mat.VecDense
	}
	photo := imgsManager.Photos[photoID]
	opticalCenter := photo.OpticalCenter()
	relevantFeats, ids := getRelevantFeatures(feat, photoID, relevantImgs)

	relevantFeatData := make([]FeatSort, len(relevantFeats), len(relevantFeats))

	featDataFiltered := make([]FeatSort, 0, len(relevantFeatData))

	depthVector1, depthVector2 := mat.NewVecDense(4, nil), mat.NewVecDense(4, nil)
	for feat2Id, feat2 := range relevantFeats {
		photo2 := imgsManager.Photos[ids[feat2Id]]
		center := triangulate(feat.X, feat.Y, feat2.X, feat2.Y, photoID, ids[feat2Id])
		center.ScaleVec(1/center.AtVec(3), center)

		if !visualHullCheck(center) {
			continue
		}

		depthVector1.SubVec(opticalCenter, center)
		depthVector2.SubVec(photo2.OpticalCenter(), center)
		depth1 := math.Sqrt(mat.Dot(depthVector1, depthVector1))
		depth2 := math.Sqrt(mat.Dot(depthVector2, depthVector2))
		relDepth := math.Abs(depth1 - depth2)
		ff := FeatSort{
			feat2, ids[feat2Id], relDepth, center,
		}
		featDataFiltered = append(featDataFiltered, ff)
	}
	sort.Slice(featDataFiltered, func(i, j int) bool {
		return featDataFiltered[i].relDepth < featDataFiltered[j].relDepth
	})

	patch := new(Patch)
	patch.Center = mat.NewVecDense(4, nil)
	patch.Normal = mat.NewVecDense(4, nil)
	patch.RefPhoto = photoID
	for _, ff := range featDataFiltered {
		patch.Center = ff.pos3d
		patch.Normal.SubVec(opticalCenter, patch.Center)
		patch.Normal.ScaleVec(1/math.Sqrt(mat.Dot(patch.Normal, patch.Normal)),
			patch.Normal)
		patch.TPhotos = constraintPhotos(patch, 0.6, relevantImgs)
		if len(patch.TPhotos) <= 1 {
			continue
		}
		optimizePatch(patch)
		patch.TPhotos = constraintPhotos(patch, 0.7, relevantImgs)
		if len(patch.TPhotos) >= 3 {
			imgsManager.Patches = append(imgsManager.Patches, patch)
			return 1
		}
	}
	return 0
}
