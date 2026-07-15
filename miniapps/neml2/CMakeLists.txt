# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

find_package(neml2 CONFIG)

if(MFEM_USE_MPI AND neml2_FOUND)
  list(APPEND MFEM_NEML2_SOURCES
    neml2.cpp
    operators.cpp
    constitutive.cpp)
  list(APPEND MFEM_NEML2_HEADERS
    operators.hpp
    constitutive.hpp)

  convert_filenames_to_full_paths(MFEM_NEML2_SOURCES)
  convert_filenames_to_full_paths(MFEM_NEML2_HEADERS)

  set(NEML2_COMMON_FILES
    EXTRA_SOURCES ${MFEM_NEML2_SOURCES}
    EXTRA_HEADERS ${MFEM_NEML2_HEADERS})

  add_mfem_miniapp(neml2
    MAIN neml2.cpp
    ${NEML2_COMMON_FILES}
    LIBRARIES mfem)

  # Link neml2 (v3 cpp-aoti runtime; drags in torch::core + nlohmann_json)
  target_link_libraries(neml2 PRIVATE neml2::aoti)
  # Ensure transitively-needed shared libs (e.g. libgomp) are kept
  target_link_options(neml2 PRIVATE -Wl,--no-as-needed)

  # Direct PETSc headers for solver introspection (SNES iteration counts) and the
  # GAMG coarse path (near-null-space). MFEM wraps PETSc types in mfem::petsc and
  # does not export the include dir, so add it explicitly.
  if(PETSC_INCLUDES)
    target_include_directories(neml2 PRIVATE ${PETSC_INCLUDES})
  endif()

  # Add parallel tests.
  if(MFEM_ENABLE_TESTING)
    add_test(NAME neml2_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:neml2>
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
