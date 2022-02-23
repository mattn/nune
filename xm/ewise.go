// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xm

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/nune/internal"
	"github.com/vorduin/nune/tensor"
	"github.com/vorduin/slices"
)

func broadShape(s1, s2 []int) []int {
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
func Ewise[T nune.Number](lhs, rhs any, f func(T, T) T) tensor.Tensor[T] {
	t1 := tensor.From[T](lhs)
	if t1.Err != nil {
		if nune.EnvConfig.Interactive {
			panic(t1.Err)
		} else {
			return t1
		}
	}

	t2 := tensor.From[T](rhs)
	if t2.Err != nil {
		if nune.EnvConfig.Interactive {
			panic(t2.Err)
		} else {
			return t2
		}
	}

	for !slices.Equal(t1.Shape(), t2.Shape()) {
		if s := broadShape(t2.Shape(), t1.Shape()); t1.Broadable(s...) && !slices.Equal(t1.Shape(), s) {
			if _, ok := lhs.(tensor.Tensor[T]); ok { // avoid wasting memory on a useless copy
				t1 = t1.Clone().Broadcast(s...)
			} else {
				t1 = t1.Broadcast(s...)
			}
		} else if s = broadShape(t1.Shape(), t2.Shape()); t2.Broadable(s...) && !slices.Equal(t2.Shape(), s) {
			if _, ok := rhs.(tensor.Tensor[T]); ok { // avoid wasting memory on a useless copy
				t2 = t2.Clone().Broadcast(s...)
			} else {
				t2 = t2.Broadcast(s...)
			}
		} else {
			if nune.EnvConfig.Interactive {
				panic(tensor.ErrNotBroadable)
			} else {
				return tensor.Tensor[T]{
					Err: tensor.ErrNotBroadable,
				}
			}
		}
	}

	res := slices.WithLen[T](t1.Numel())

	internal.HandleEwise(t1.Ravel(), t2.Ravel(), res, f, 4)

	return tensor.From[T](res).Reshape(t1.Shape()...)
}

// Add takes two values and performs elementwise addition over their elements.
func Add[T nune.Number](lhs, rhs any) tensor.Tensor[T] {
	return Ewise(lhs, rhs, func(x, y T) T {
		return x + y
	})
}