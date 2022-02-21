// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

import "errors"

// List of errors.
var (
	// ErrAxisBounds occurs when an axis is out of
	// (0, rank) bounds.
	errAxisBounds = errors.New("nune/tensor: axis out of bounds")
)

// assertAxisBounds makes sure an axis is strictly positive
// and is less than the Tensor's rank.
func assertAxisBounds(axis, rank int) {
	if axis < 0 || axis > rank {
		panic(errAxisBounds)
	}
}
