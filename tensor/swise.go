// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	// "github.com/vorduin/nune"
	"github.com/vorduin/nune/internal"
)

// Swise performs a reduction operation over all elements in the Tensor.
// The reduction operation must be able to generelize and parallelize
// since the operation might be multi-threaded if the Tensor is big enough,
// unless explicitely disabled in Nune's environment configuration.
func (t Tensor[T]) Swise(f func([]T) T) Tensor[T] {
	var res T
	internal.HandleSwise(t.Ravel(), &res, f, 1)
	
	return Tensor[T]{
		data: []T{res},
	}
}

// Min returns the minimum value of all elements in the Tensor.
func (t Tensor[T]) Min() Tensor[T] {
	return t.Swise(func(s []T) T {
		min := s[0]
		for i := 1; i < len(s); i++ {
			if s[i] < min {
				min = s[i]
			}
		}
		return min
	})
}

// Max returns the maximum value of all elements in the Tensor.
func (t Tensor[T]) Max() Tensor[T] {
	return t.Swise(func(s []T) T {
		max := s[0]
		for i := 1; i < len(s); i++ {
			if s[i] > max {
				max = s[i]
			}
		}
		return max
	})
}

// Mean returns the mean value of all elements in the Tensor.
func (t Tensor[T]) Mean() Tensor[T] {
	return t.Swise(func(s []T) T {
		var sum T
		for i := 0; i < len(s); i++ {
			sum += s[i]
		}
		return sum / T(len(s))
	})
}

// Sum returns the sum of all elements in the Tensor.
func (t Tensor[T]) Sum() Tensor[T] {
	return t.Swise(func(s []T) T {
		var sum T
		for i := 0; i < len(s); i++ {
			sum += s[i]
		}
		return sum
	})
}

// Prod returns the product of all elements in the Tensor.
func (t Tensor[T]) Prod() Tensor[T] {
	return t.Swise(func(s []T) T {
		var prod T = 1
		for i := 0; i < len(s); i++ {
			prod *= s[i]
		}
		return prod
	})
}