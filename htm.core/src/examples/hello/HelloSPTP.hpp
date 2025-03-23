// header file for HelloSPTP
#ifndef NTA_EXAMPLES_HELLO_SPTP_
#define NTA_EXAMPLES_HELLO_SPTP_

#include <htm/types/Types.hpp>
#include <htm/os/Timer.hpp>

namespace examples {

using htm::Real64;
using htm::UInt;
using htm::Timer;

class BenchmarkHelloSPTP {

public:	
  Real64 run(
    UInt EPOCHS = 5000,
    bool useSPlocal=true, //can toggle which (long running) components are tested, default all
    bool useSPglobal=true,
    bool useTM=true,
    const UInt COLS = 2048, // number of columns in SP, TP
    const UInt DIM_INPUT = 1000,
    const UInt CELLS = 8 // cells per column in TP
  );

  //timers
  Timer tInit, tAll, tRng, tEnc, tSPloc, tSPglob, tTM, tAnLikelihood;
};

} //-ns
#endif //NTA_EXAMPLES_HELLO_SPTP_
