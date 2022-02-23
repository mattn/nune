// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import "github.com/vorduin/slices"

// configStride returns the corresponding stride to the given shape.
func configStride(shape []int) []int {
	if len(shape) != 0 {
		stride := slices.WithLen[int](len(shape))
		stride[0] = slices.Prod(shape[1:])

		for i := 1; i < len(shape); i++ {
			stride[i] = stride[i-1] / shape[i]
		}

		return stride
	}

	return nil
}
