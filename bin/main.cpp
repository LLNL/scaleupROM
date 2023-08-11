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
   const char *output_file = "";
   const char *forced_input = "";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input file to use.");
   args.AddOption(&output_file, "-o", "--output", "For single run, save comparison result to output file.");
   args.AddOption(&forced_input, "-f", "--forced-input",
      "Input options to overwrite. In the format of 'key1=value1:key2=value2:...'");
   args.ParseCheck();
   config = InputParser(input_file, forced_input);

   std::string mode = config.GetOption<std::string>("main/mode", "run_example");

   if (mode == "run_example")
   { if (rank == 0) RunExample(); }
   else 
   {
      if (mode == "sample_generation") GenerateSamples(MPI_COMM_WORLD);
      else if (mode == "build_rom")    BuildROM(MPI_COMM_WORLD);
      else if (mode == "train_rom")    TrainROM(MPI_COMM_WORLD);
      else if (mode == "single_run")   double dump = SingleRun(output_file);
      else
      {
         if (rank == 0) printf("Unknown mode %s!\n", mode.c_str());
         exit(-1);
      }
   }
   
   MPI_Finalize();
}