# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc.
# modified 4/4/2022 - newer version
# modified 12/12/2024 - use FetchContent, David Keeney dkeeney@gmail.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# -----------------------------------------------------------------------------

# Download Eigen from GitHub archive
# Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, 
# and related algorithms.
# For more information go to http://eigen.tuxfamily.org/.
# For repository, go to https://gitlab.com/libeigen/eigen.
#
# This file downloads eigen.	
# No build. This is a header only package
include(FetchContent)

set(dependency_url "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz")
set(local_override "${CMAKE_SOURCE_DIR}/build/Thirdparty/eigen")

# Check if local path exists and if so, use it as-is.
if(EXISTS ${local_override})
    message(STATUS "  Obtaining Eigen from local override: ${local_override}")
    FetchContent_Populate(
        eigen
        SOURCE_DIR ${local_override}
		QUIET
    )
else()
    message(STATUS "  Obtaining Eigen from: ${dependency_url}  HEADER_ONLY")
    FetchContent_Populate(
        eigen
        URL ${dependency_url}
		QUIET
    )
endif()

# This does have a CMakeLists.txt.  By default tests are turned off.
# It does need to run CMakeLists to configure eigen.h

# to access #include "eigen"
set(eigen_INCLUDE_DIR "${eigen_SOURCE_DIR}")
