#!/bin/bash

mkdir -p build
cd build
cmake ../native
make install # installs into source directory
cd -

dotnet build
dotnet test