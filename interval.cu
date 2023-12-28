#include "interval.h"

// Static member initialization
const interval interval::empty = interval(+FLT_MAX, -FLT_MAX);
const interval interval::universe = interval(-FLT_MAX, +FLT_MAX);
