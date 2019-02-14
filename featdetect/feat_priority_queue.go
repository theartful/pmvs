package featdetect

import "math"

// A FeatPriorityQueue implements heap.Interface and holds features.
type FeatPriorityQueue []*Feature

// Push : Appends a new element
// Used to implement the heap interface
func (pq *FeatPriorityQueue) Push(x interface{}) {
	item := x.(*Feature)
	*pq = append(*pq, item)
}

// Pop : Removes and returns the last element
// Used to implement the heap interface
func (pq *FeatPriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

func (pq FeatPriorityQueue) Len() int { return len(pq) }

func (pq FeatPriorityQueue) Less(i, j int) bool {
	return math.Abs(float64(pq[i].Response)) <
		math.Abs(float64(pq[j].Response))
}

func (pq FeatPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}
