#ifndef __SRC_MACRO_H__
#define __SRC_MACRO_H__

#include <string.h>
#include <assert.h>
#include <iostream>

/* Machine zero of double precision*/
#define NZERO 1.0e-13

/* Transformation between 2D dense storageGlob index and 1D banded storageGlob*/
#define DenseToBand1D(r, c, kl, ku) ((kl + ku + 1) * c + ku + r - c)
#define DenseToBand2D(r, c, ku) (ku + r - c)

/* Coordinate transformation */
#define TC(x, y, ld) ((y) * (ld) + (x))

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#endif