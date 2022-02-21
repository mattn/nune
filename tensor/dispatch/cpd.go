// Copyright © The Nune Author. All rights reserved.
// Use of this source logic, code, or documentation is governed by
// an MIT-style license that can be found in the LICENSE file.

package dispatch

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// CPD is the Central Processing Dispatch.
// It defines how the Tensor's data is stored, processed, and handled on CPU.
// This is Nune's default dispatch.
type CPD[T nune.Number] struct {
	data []T
}

// Dump sets the Tensor's data buffer on CPU to
// the given buffer.
func (c CPD[T]) Dump(data []T) {
	c.data = data
}

// Load loads the Tensor's data buffer from RAM.
func (c CPD[T]) Load() []T {
	return c.data
}

// Index returns the value at the given index from
// the Tensor's 1-dimensional data buffer on RAM.
func (c CPD[T]) Index(idx int) T {
	return c.data[idx]
}

// SetIndex sets the index of the Tensor's 1-dimensional
// data buffer on RAM to the given value.
func (c CPD[T]) SetIndex(idx int, value T) {
	c.data[idx] = value
}

// Slice returns a new Dispatch holding a slice of the given bounds from
// the Tensor's 1-dimensional data buffer on RAM.
func (c *CPD[T]) Slice(start, end int) Dispatcher[T] {
	cpd := new(CPD[T])
	cpd.data = c.data[start:end]
	return any(cpd).(Dispatcher[T])
}

// SetSlice sets the slice from the given bounds of
// the Tensor's 1-dimensional data buffer on RAM to the given slice.
func (c CPD[T]) SetSlice(start, end int, s []T) {
	idx := 0
	for i := start; i < end; i++ {
		c.data[i] = s[idx]
		idx++
	}
}

// Copy copies the CPD's underlying data into a new CPD
// and returns it.
func (c CPD[T]) Copy() Dispatcher[T] {
	cpd := new(CPD[T])
	cpd.data = slices.Copy(c.data)
	return any(cpd).(Dispatcher[T])
}

// Cast returns a pointer to a new CPD casted as any,
// with the underlying data casted to the type of the given value.
func (c CPD[T]) Cast(value any) any {
	switch value.(type) {
	case int:
		return cpdCast[int](c)
	case int8:
		return cpdCast[int8](c)
	case int16:
		return cpdCast[int16](c)
	case int32:
		return cpdCast[int32](c)
	case int64:
		return cpdCast[int64](c)
	case uint:
		return cpdCast[uint](c)
	case uint8:
		return cpdCast[uint8](c)
	case uint16:
		return cpdCast[uint16](c)
	case uint32:
		return cpdCast[uint32](c)
	case uint64:
		return cpdCast[uint64](c)
	case float32:
		return cpdCast[float32](c)
	case float64:
		return cpdCast[float64](c)
	default:
		return nil
	}
}

// cpdCast creates a new CPD with a different underlying numeric type.
func cpdCast[T nune.Number, U nune.Number](c CPD[U]) *CPD[T] {
	cast := new(CPD[T])
	cast.data = numericCast[T](c.data)
	return cast
}

// numericCast casts a numeric type to another numeric type.
func numericCast[T, U nune.Number](s []U) []T {
	ns := slices.WithLen[T](len(s))
	for i := 0; i < len(s); i++ {
		ns[i] = T(s[i])
	}

	return ns
}