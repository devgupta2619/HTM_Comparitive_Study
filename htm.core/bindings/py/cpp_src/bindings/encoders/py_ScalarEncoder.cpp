/* ----------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, David McDougall
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <sstream>
namespace py = pybind11;

#include <htm/encoders/ScalarEncoder.hpp>
#include <htm/types/Sdr.hpp>

namespace htm_ext
{
  using namespace htm;

  void init_ScalarEncoder(py::module& m)
  {
    py::class_<ScalarEncoderParameters> py_ScalarEncParams(m, "ScalarEncoderParameters",
        R"(

The following four (4) members define the total number of bits in the output:
     size,
     radius,
     category,
     resolution.

These are mutually exclusive and only one of them should be non-zero when
constructing the encoder.)");

    py_ScalarEncParams.def(py::init<>(), R"()");

    py_ScalarEncParams.def_readwrite("minimum", &ScalarEncoderParameters::minimum,
R"(This defines the range of the input signal. These endpoints are inclusive.)");

    py_ScalarEncParams.def_readwrite("maximum", &ScalarEncoderParameters::maximum,
R"(This defines the range of the input signal. These endpoints are inclusive.)");

    py_ScalarEncParams.def_readwrite("clipInput", &ScalarEncoderParameters::clipInput,
R"(This determines whether to allow input values outside the
range [minimum, maximum].
If true, the input will be clipped into the range [minimum, maximum].
If false, inputs outside of the range will raise an error.)");

    py_ScalarEncParams.def_readwrite("periodic", &ScalarEncoderParameters::periodic,
R"(This controls what happens near the edges of the input range.

If true, then the minimum & maximum input values are adjacent and the first and
last bits of the output SDR are also adjacent.  The contiguous block of 1's
wraps around the end back to the beginning.

If false, then minimum & maximum input values are the endpoints of the input
range, are not adjacent, and activity does not wrap around.)");

    py_ScalarEncParams.def_readwrite("category", &ScalarEncoderParameters::category,
R"(This means that the inputs are enumerated categories.
If true then this encoder will only encode unsigned integers, and all inputs
will have unique / non-overlapping representations.)");

    py_ScalarEncParams.def_readwrite("activeBits", &ScalarEncoderParameters::activeBits,
R"(This is the number of true bits in the encoded output SDR. The output
encodings will have a contiguous block of this many 1's.)");

    py_ScalarEncParams.def_readwrite("sparsity", &ScalarEncoderParameters::sparsity,
R"(This is an alternative way to specify the the number of active bits.
Sparsity requires that the size to also be specified.
Specify only one of: activeBits or sparsity.)");

    py_ScalarEncParams.def_readwrite("size", &ScalarEncoderParameters::size,
R"(This is the total number of bits in the encoded output SDR.)");

    py_ScalarEncParams.def_readwrite("radius", &ScalarEncoderParameters::radius,
R"(Two inputs separated by more than the radius have non-overlapping
representations. Two inputs separated by less than the radius will in general
overlap in at least some of their bits. You can think of this as the radius of
the input.)");

    py_ScalarEncParams.def_readwrite("resolution", &ScalarEncoderParameters::resolution,
R"(Two inputs separated by greater than, or equal to the resolution are guaranteed
to have different representations.)");


    py::class_<ScalarEncoder> py_ScalarEnc(m, "ScalarEncoder",
R"(Encodes a real number as a contiguous block of 1's.

The ScalarEncoder encodes a numeric (floating point) value into an array of
bits. The output is 0's except for a contiguous block of 1's. The location of
this contiguous block varies continuously with the input value.

To inspect this run:
$ python -m htm.examples.encoders.scalar_encoder --help)");

    py_ScalarEnc.def(py::init<>(), R"( For use with loadFromFile. )");
    py_ScalarEnc.def(py::init<ScalarEncoderParameters&>(), R"()");
    py_ScalarEnc.def_property_readonly("parameters",
        [](const ScalarEncoder &self) { return self.parameters; },
R"(Contains the parameter structure which this encoder uses internally. All
fields are filled in automatically.)");

    py_ScalarEnc.def_property_readonly("dimensions",
        [](const ScalarEncoder &self) { return self.dimensions; });
    py_ScalarEnc.def_property_readonly("size",
        [](const ScalarEncoder &self) { return self.size; });

    py_ScalarEnc.def("encode", &ScalarEncoder::encode, R"()");

    py_ScalarEnc.def("encode", [](ScalarEncoder &self, htm::Real64 value) {
        auto output = new SDR( self.dimensions );
        self.encode( value, *output );
        return output; },
R"()");

// Serialization
// loadFromString
    py_ScalarEnc.def("loadFromString", [](ScalarEncoder& self, const py::bytes& inString) {
      std::stringstream inStream(inString.cast<std::string>());
      self.load(inStream, JSON);
    });

    // writeToString
    py_ScalarEnc.def("writeToString", [](const ScalarEncoder& self) {
      std::ostringstream os;
      os.flags(std::ios_base::scientific);
      os.precision(std::numeric_limits<double>::digits10 + 1);
      self.save(os, JSON); // see serialization in bindings for SP, py_SpatialPooler.cpp for explanation
      return py::bytes( os.str() );
   });

  // pickle
  py_ScalarEnc.def(py::pickle( 
  [](const ScalarEncoder& enc) // __getstate__
  {
    std::stringstream ss;
    enc.save(ss);

  /* The values in stringstream are binary so pickle will get confused
   * trying to treat it as utf8 if you just return ss.str().
   * So we must treat it as py::bytes.  Some characters could be null values.
   */
    return py::bytes( ss.str() );
  },

  [](py::bytes &s)   // __setstate__
  {
  /* pybind11 will pass in the bytes array without conversion.
   * so we should be able to just create a string to initalize the stringstream.
   */
    std::stringstream ss( s.cast<std::string>() );
    std::unique_ptr<ScalarEncoder> enc(new ScalarEncoder());
    enc->load(ss);

  /*
   * The __setstate__ part of the py::pickle() is actually a py::init() with some options.
   * So the return value can be the object returned by value, by pointer,
   * or by container (meaning a unique_ptr). SP has a problem with the copy constructor
   * and pointers have problems knowing who the owner is so lets use unique_ptr.
   * See: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#custom-constructors
   */
    return enc;
  }));

// loadFromFile
//    
   py_ScalarEnc.def("saveToFile",
                     static_cast<void (htm::ScalarEncoder::*)(std::string, std::string) const>(&htm::ScalarEncoder::saveToFile), 
                     py::arg("file"), py::arg("fmt") = "BINARY",
R"(Serializes object to file. file: filename to write to.  fmt: format, one of 'BINARY', 'PORTABLE', 'JSON', or 'XML')");

   py_ScalarEnc.def("loadFromFile",    static_cast<void (htm::ScalarEncoder::*)(std::string, std::string)>(&htm::ScalarEncoder::loadFromFile), 
                     py::arg("file"), py::arg("fmt") = "BINARY",
R"(Deserializes object from file. file: filename to read from.  fmt: format recorded by saveToFile(). )");


  }
}
