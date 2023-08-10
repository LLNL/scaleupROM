#include "main_workflow.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const char *input_file = "test.input";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input file to use.");
   args.ParseCheck();
   config = InputParser(input_file);

   std::string mode = config.GetOption<std::string>("main/mode", "run_example");

   if (mode == "run_example")
   { if (rank == 0) RunExample(); }
   else 
   {
      if (mode == "sample_generation") GenerateSamples(MPI_COMM_WORLD);
      else if (mode == "build_rom")    BuildROM(MPI_COMM_WORLD);
      else if (mode == "train_rom")    TrainROM(MPI_COMM_WORLD);
      else if (mode == "single_run")   double dump = SingleRun();
      else
      {
         if (rank == 0) printf("Unknown mode %s!\n", mode.c_str());
         exit(-1);
      }
   }
   
   MPI_Finalize();
}