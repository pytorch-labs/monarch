/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include "rdmaxcel.h"

int main() {
  void* func_ptr = (void*)&cu_db_ring;
  printf("cu_db_ring function address: %p\n", func_ptr);
  return 0;
}
