// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package tensor

import "github.com/vorduin/slices"

// configStrides returns the corresponding strides to the given shape.
func configStrides(shape []int) []int {
	if len(shape) != 0 {
		strides := slices.WithLen[int](len(shape))
		strides[0] = slices.Prod(shape[1:])

		for i := 1; i < len(shape); i++ {
			strides[i] = strides[i-1] / shape[i]
		}

		return strides
	}

	return nil
}
