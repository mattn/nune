// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/nune/internal"
	"github.com/vorduin/slices"
)

// Ewise performs an elementwise operation over the elements of the Tensor
// and a broadcastable numerical other.
func (t Tensor[T]) Ewise(other any, f func(T, T) T) Tensor[T] {
	t2 := From[T](other)
	if t2.Err != nil {
		if nune.EnvConfig.Interactive {
			panic(t.Err)
		} else {
			t.Err = t2.Err
			return t
		}
	}

	for !slices.Equal(t.shape, t2.shape) {
		if t2.Broadable(t.shape...) {
			if _, ok := other.(Tensor[T]); ok { // avoid wasting memory on a useless clone
				t2 = t2.Clone().Broadcast(t.shape...)
			} else {
				t2 = t2.Broadcast(t.shape...)
			}
		} else {
			if nune.EnvConfig.Interactive {
				panic(ErrNotBroadable)
			} else {
				t.Err = ErrNotBroadable
				return t
			}
		}
	}

	internal.HandleEwise(t.Ravel(), t2.Ravel(), t.Ravel(), f, 4)

	return t
}

// Add takes a value and performs elementwise addition
// with the elements of the current Tensor.
func (t Tensor[T]) Add(other any) Tensor[T] {
	return t.Ewise(other, func(x, y T) T {
		return x + y
	})
}

// Sub takes a value and performs elementwise subtraction
// with the elements of the current Tensor.
func (t Tensor[T]) Sub(other any) Tensor[T] {
	return t.Ewise(other, func(x, y T) T {
		return x - y
	})
}

// Mul takes a value and performs elementwise multiplication
// with the elements of the current Tensor.
func (t Tensor[T]) Mul(other any) Tensor[T] {
	return t.Ewise(other, func(x, y T) T {
		return x * y
	})
}

// Div takes a value and performs elementwise division
// with the elements of the current Tensor.
func (t Tensor[T]) Div(other any) Tensor[T] {
	return t.Ewise(other, func(x, y T) T {
		return x / y
	})
}