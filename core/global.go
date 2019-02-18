package core

// global variables in this case are ok since:
// 	1 - the package handles only one dataset at a time
// 	2 - most of the functions use the images manager

const (
	cosMinAngle   = 0.9396926207859084 //math.Cos(20 * math.Pi / 180)
	cosMaxAngle   = 0.5                //math.Cos(60 * math.Pi / 180)
	featMaxDist   = 2.0
	patchGridSize = 5
)

var (
	imgsManager *ImagesManager
)
