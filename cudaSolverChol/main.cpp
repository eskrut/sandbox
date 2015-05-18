#include "sbfMesh.h"
#include "sbfPropertiesSet.h"
#include "sbfStiffMatrixBand.h"
#include "sbfGroupFilter.h"
#include "sbfNodeGroup.h"
#include <memory>

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

#include <helper_cuda.h>

int main(int argc, char **argv)
{
    float numScale = 4.5;
    std::unique_ptr<sbfMesh> mesh_up(sbfMesh::makeBlock(1000, 10, 10, 30*numScale, 4*numScale, 4*numScale));
//    std::unique_ptr<sbfMesh> mesh_up(sbfMesh::makeBlock(1, 1, 1, 1, 1, 1));
    sbfMesh *mesh = mesh_up.get();
    mesh->setMtr(1);
    report("Optimizing node numbering");
    mesh->optimizeNodesNumbering(RenumberOptimizationType::SIMPLE, false);
    report("Num nodes:", mesh->numNodes());

    NodesData<> displ("displ", mesh), force("force", mesh);
    force.null();

    std::unique_ptr<sbfPropertiesSet> props_up(std::make_unique<sbfPropertiesSet>());
    sbfPropertiesSet *props = props_up.get();
    props->addMaterial(sbfMaterialProperties::makeMPropertiesSteel());

    report("Constructing stiff");
    std::unique_ptr<sbfStiffMatrixBand<3>> stiff_up(std::make_unique<sbfStiffMatrixBand<3>>(mesh, props, MatrixType::FULL_MATRIX));
    sbfStiffMatrixBand<3> *stiff = stiff_up.get();

    report("Computing stiff");
    stiff->compute();

    sbfGroupFilter filtLock, filtLoad;
    filtLock.setCrdXF(mesh->minX()-0.001, mesh->minX()+0.001);
    filtLoad.setCrdXF(mesh->maxX()-0.001, mesh->maxX()+0.001);

    mesh->addNodeGroup(filtLock);
    mesh->addNodeGroup(filtLoad);
    mesh->processNodeGroups();

    auto listLock = mesh->nodeGroup(0)->nodeIndList();
    auto listLoad = mesh->nodeGroup(1)->nodeIndList();

    for(auto i : listLoad) force(i, 0) = 1.0/listLoad.size();

    report("Locking DOFs");
    for(auto i : listLock) stiff->lockDof(i, 0, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
    stiff->lockDof(mesh->nodeAt(0, 0, 0), 1, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
    stiff->lockDof(mesh->nodeAt(0, 0, 0), 2, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
    stiff->lockDof(mesh->nodeAt(0, mesh->maxY(), 0), 2, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
    stiff->lockDof(mesh->nodeAt(0, 0, mesh->maxZ()), 1, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);

    {
        displ.null();
        report("Making chol");
        std::unique_ptr<sbfStiffMatrix> chol_up(stiff->createChol());
        sbfStiffMatrix *chol = chol_up.get();

        if ( !chol->isValid() ) report.error("Chol factor is not valid");

        report("Solving");
        chol->solve_L_LT_u_eq_f(displ.data(), force.data());

        double av = 0;
        for(auto i : listLoad)
            av += displ(i, 0);
        av /= listLoad.size();

        report("Calc: ", av, "\nexp: ", mesh->maxX()*1.0/props->material(0)->propertyTable("elastic module")->curValue()/mesh->maxX()/mesh->maxZ());

    }

    {
        displ.null();
        report("Making ldlt");
        std::unique_ptr<sbfStiffMatrix> ldlt_up(stiff->createLDLT());
        sbfStiffMatrix *ldlt = ldlt_up.get();

        if ( !ldlt->isValid() ) report.error("Chol factor is not valid");

        report("Solving");
        ldlt->solve_L_D_LT_u_eq_f(displ.data(), force.data());

        double av = 0;
        for(auto i : listLoad)
            av += displ(i, 0);
        av /= listLoad.size();

        report("Calc: ", av, "\nexp: ", mesh->maxX()*1.0/props->material(0)->propertyTable("elastic module")->curValue()/mesh->maxX()/mesh->maxZ());

    }

    {
        displ.null();
        /* This will pick the best possible CUDA capable device */
        cudaDeviceProp deviceProp;
        int devID = findCudaDevice(argc, (const char **)argv);
        printf("GPU selected Device ID = %d \n", devID);

        if (devID < 0)
        {
            printf("Invalid GPU device %d selected,  exiting...\n", devID);
            exit(EXIT_SUCCESS);
        }

        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

        /* Statistics about the GPU device */
        printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
               deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

        int version = (deviceProp.major * 0x10 + deviceProp.minor);

        if (version < 0x11)
        {
            printf("requires a minimum CUDA compute 1.1 capability\n");

            // cudaDeviceReset causes the driver to clean up all state. While
            // not mandatory in normal operation, it is good practice.  It is also
            // needed to ensure correct operation when the application is being
            // profiled. Calling cudaDeviceReset causes all profile data to be
            // flushed before the application exits
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
        }

        /* Create CUBLAS context */
        cublasHandle_t cublasHandle = 0;
        cublasStatus_t cublasStatus;
        cublasStatus = cublasCreate(&cublasHandle);

        checkCudaErrors(cublasStatus);

        /* Create CUSPARSE context */
        cusparseHandle_t cusparseHandle = 0;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);

        checkCudaErrors(cusparseStatus);

        /* Description of the A matrix*/
        cusparseMatDescr_t descr = 0;
        cusparseStatus = cusparseCreateMatDescr(&descr);

        checkCudaErrors(cusparseStatus);

        /* Define the properties of the matrix */
        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

        std::vector<int> colInds, rowInds;
        std::unique_ptr<sbfMatrixIterator> iter_up(stiff->createIterator());
        sbfMatrixIterator *iter = iter_up.get();
        int ct = 0;
        for(; ct < mesh->numNodes(); ++ct) {
            iter->setToRow(ct);
            rowInds.push_back(ct);
            while (iter->isValid()) {
                colInds.push_back(iter->column());
                iter->next();
            }
        }
        rowInds.push_back(ct);

        int *d_col, *d_row;
        double *d_val, *d_x, *d_y, *d_r;

        /* Allocate required memory */
        checkCudaErrors(cudaMalloc((void **)&d_col, colInds.size()*sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_row, rowInds.size()*sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_val, stiff->numDof()*stiff->numDof()*colInds.size()*sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_x, mesh->numNodes()*3*sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_y, mesh->numNodes()*3*sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_r, mesh->numNodes()*3*sizeof(double)));

        cudaMemcpy(d_col, colInds.data(), colInds.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row, rowInds.data(), rowInds.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, stiff->data(), stiff->numDof()*stiff->numDof()*colInds.size()*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, displ.data(), mesh->numNodes()*3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, force.data(), mesh->numNodes()*3*sizeof(double), cudaMemcpyHostToDevice);
    }

    return 0;
}

