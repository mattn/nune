// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package nune

// Number is the set of all numeric types and their supersets.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}
