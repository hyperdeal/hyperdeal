// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the hyper.deal authors
//
// This file is part of the hyper.deal library.
//
// The hyper.deal library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hyper.deal.
//
// ---------------------------------------------------------------------


// Test hyperdeal::Utilities::decompose.

#include <hyper.deal/base/utilities.h>

#include "../tests.h"

int
main()
{
  initlog();

  for (unsigned int i = 1; i <= 56 /*SuperMUC - Phase 2*/; i++)
    {
      const auto result = hyperdeal::Utilities::decompose(i);
      deallog << i << " (" << result.first << " " << result.second << ")"
              << std::endl;
    }
}
