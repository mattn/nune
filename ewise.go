// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import (
	"github.com/vorduin/slices"
)

// midwayBroadcast adjusts the first shape so that the second shape might
// be broadcastable to it. For example, for the shape [4, 1] to be
// broadcastable with the shape [3], this function returns that
// the broadcasting shape should be [4, 3].
func midwayBroadcast(s1, s2 []int) []int {
	var s []int

	if len(s1) < len(s2) {
		s = slices.WithLen[int](len(s2))
		for i := 0; i < len(s2)-len(s1); i++ {
			s[i] = 1
		}
		copy(s[len(s2)-len(s1):], s1)
	} else {
		s = s1
	}

	for i := 0; i < len(s2); i++ {
		if s[i] != s2[i] && s[i] == 1 {
			s[i] = s2[i]
		}
	}

	return s
}

// Ewise performs an elementwise operation over the elements of the Tensor
// and a broadcastable numerical other.
func Ewise[T Number](lhs, rhs any, f func(T, T) T) Tensor[T] {
	t1 := From[T](lhs)
	if t1.Err != nil {
		if EnvConfig.Interactive {
			panic(t1.Err)
		} else {
			return t1
		}
	}

	t2 := From[T](rhs)
	if t2.Err != nil {
		if EnvConfig.Interactive {
			panic(t2.Err)
		} else {
			return t2
		}
	}

	for !slices.Equal(t1.Shape(), t2.Shape()) {
		if s := midwayBroadcast(t2.Shape(), t1.Shape()); t1.Broadable(s...) && !slices.Equal(t1.Shape(), s) {
			if _, ok := lhs.(Tensor[T]); ok { // avoid wasting memory on a useless copy
				t1 = t1.Clone().Broadcast(s...)
			} else {
				t1 = t1.Broadcast(s...)
			}
		} else if s = midwayBroadcast(t1.Shape(), t2.Shape()); t2.Broadable(s...) && !slices.Equal(t2.Shape(), s) {
			if _, ok := rhs.(Tensor[T]); ok { // avoid wasting memory on a useless copy
				t2 = t2.Clone().Broadcast(s...)
			} else {
				t2 = t2.Broadcast(s...)
			}
		} else {
			if EnvConfig.Interactive {
				panic(ErrNotBroadable)
			} else {
				return Tensor[T]{
					Err: ErrNotBroadable,
				}
			}
		}
	}

	res := slices.WithLen[T](t1.Numel())

	handleEwise(t1.Ravel(), t2.Ravel(), res, f, 4)

	return From[T](res).Reshape(t1.Shape()...)
}

// Ewise performs an elementwise operation over the elements of the Tensor
// and a broadcastable numerical other.
func (t Tensor[T]) Ewise(other any, f func(T, T) T) Tensor[T] {
	if t.Err != nil {
		if EnvConfig.Interactive {
			panic(t.Err)
		} else {
			return t
		}
	}

	t2 := From[T](other)
	if t2.Err != nil {
		if EnvConfig.Interactive {
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
			if EnvConfig.Interactive {
				panic(ErrNotBroadable)
			} else {
				t.Err = ErrNotBroadable
				return t
			}
		}
	}

	handleEwise(t.Ravel(), t2.Ravel(), t.Ravel(), f, 4)

	return t
}

// Add takes two values and performs elementwise addition over their elements.
func Add[T Number](lhs, rhs any) Tensor[T] {
	return Ewise(lhs, rhs, func(x, y T) T {
		return x + y
	})
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
