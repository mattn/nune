// Copyright © The Nune Author. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

import "errors"

// List of errors.
var (
	// ErrBadShape occurs when a shape is nil or a has axes whose
	// dimensions are less than or equal to zero.
	ErrBadShape = errors.New("nune: received a bad shape")

	// ErrBadStep occurs when a step size is null or
	// is opposite to the inverval's order.
	ErrBadStep = errors.New("nune: received a bad step size")

	// ErrNotBroadable occurs when a Tensor could not be broadcast
	// to a provided shape.
	ErrNotBroadable = errors.New("nune: could not broadcast tensor to shape")

	// ErrBadInterval occurs when a null interval, or a descending
	// interval, or an interval that doesn't fall within the allowed limits
	// is provided to a function like range or slice.
	ErrBadInterval = errors.New("nune: received a bad interval")

	// ErrUnwrapBacking occurs when a backing could not be
	// unwrapped into a 1-dimensional numeric buffer in order
	// to create a Tensor.
	ErrUnwrapBacking = errors.New("nune: could not unwrap backing to Tensor")

	// ErrArgsBounds occurs when a function receives more arguments
	// than it should.
	ErrArgsBounds = errors.New("nune: received more arguments than allowed")

	// ErrAxisBounds occurs when an axis is out of
	// (0, rank) bounds.
	ErrAxisBounds = errors.New("nune: axis out of bounds")

	// ErrStorageDump occurs when the Assign method fails to dump
	// the given data to the Tensor's storage.
	ErrStorageDump = errors.New("nune: could not dump data buffer to storage")
)

// verifyGoodShape makes sure a shape isn't empty,
// and that none of the shapes axes's dimensions
// are less than or equal to zero, and panics otherwise.
func verifyGoodShape(s ...int) error {
	if len(s) == 0 {
		return ErrBadShape
	}

	for _, a := range s {
		if a <= 0 {
			return ErrBadShape
		}
	}

	return nil
}

// verifyGoodStep makes sure a step size isn't null,
// and whose sign matches the interval's order.
func verifyGoodStep(s, start, end int) error {
	if s == 0 {
		return ErrBadStep
	} else if s > 0 && end < start || s < 0 && end > start {
		return ErrBadStep
	}
	return nil
}

// verifyGoodInterval makes sure the interval is not null,
// is ascending, and falls within the given limits, inclusive.
// A value of nil for min or max means there is no limit.
func verifyGoodInterval(start, end int, limits ...[2]int) error {
	err := verifyArgsBounds(len(limits), 1)
	if err != nil {
		return err
	}

	if start >= end {
		return ErrBadInterval
	}

	if len(limits) == 1 {
		if start < limits[0][0] || end > limits[0][1] {
			return ErrBadInterval
		}
	}

	return nil
}

// verifyArgsBounds makes sure the number of arguments
// doesn't exceed the number of allowed arguments.
func verifyArgsBounds(nargs, max int) error {
	if nargs > max {
		return ErrArgsBounds
	}
	return nil
}

// verifyAxisBounds makes sure an axis is strictly positive
// and is less than the Tensor's rank.
func verifyAxisBounds(axis, rank int) error {
	if axis < 0 || axis > rank {
		return ErrAxisBounds
	}
	return nil
}
