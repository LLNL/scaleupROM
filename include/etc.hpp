// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef ETC_HPP
#define ETC_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

double UniformRandom();
int UniformRandom(const int &min, const int &max);

template <typename T>
inline void DeletePointers(Array<T*> &ptr_array)
{ for (int k = 0; k < ptr_array.Size(); k++) delete ptr_array[k]; }

template <typename T>
inline void DeletePointers(Array2D<T*> &ptr_array)
{
    for (int i = 0; i < ptr_array.NumRows(); i++)
        for (int j = 0; j < ptr_array.NumCols(); j++)
            delete ptr_array(i,j);
}

bool FileExists(const std::string& name);

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

class TimeProfiler
{
private:
    Array<StopWatch *> timers;

    Array<int> calls;
    std::vector<std::string> names;
    std::unordered_map<std::string, int> indices;

    const MPI_Comm comm;
    int rank;

public:
    TimeProfiler(MPI_Comm comm_=MPI_COMM_WORLD)
        : comm(comm_), timers(0)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        calls.SetSize(0);
        names.clear();
        indices.clear();
    }

    virtual ~TimeProfiler()
    {
        DeletePointers(timers);
    }

    void Start(const std::string &name)
    {
        if (!indices.count(name))
        {
            indices[name] = timers.Size();
            timers.Append(new StopWatch);
            names.push_back(name);
            calls.Append(0);
        }

        assert(indices.count(name));
        int idx = indices[name];
        timers[idx]->Start();
    }

    void Stop(const std::string &name)
    {
        assert(indices.count(name));
        int idx = indices[name];
        timers[idx]->Stop();
        calls[idx] += 1;
    }

    void Print(const std::string &title)
    {
        int nfunc = timers.Size();
        Array<double> times(nfunc);
        for (int k = 0; k < nfunc; k++)
            times[k] = timers[k]->RealTime();

        MPI_Reduce(MPI_IN_PLACE, times.GetData(), nfunc, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, calls.GetData(), nfunc, MPI_INT, MPI_SUM, 0, comm);

        if (rank == 0)
        {
            printf((title + "\n").c_str());

            std::string line = std::string(100, '=');
            line += "\n";
            printf(line.c_str());
            printf("%20s\t%20s\t%20s\t%20s\n", "Function", "Total time", "Calls", "Time per call");
            for (int k = 0; k < nfunc; k++)
            {
                printf("%20s\t%20.5e\t%20d\t%20.5e\n", names[k].c_str(), times[k], calls[k], times[k] / calls[k]);
            }
            printf(line.c_str());
        }
    }
};

#endif
