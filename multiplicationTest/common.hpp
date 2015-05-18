#ifndef COMMON_HPP
#define COMMON_HPP

#include "sbfMesh.h"
#include "sbfNodeGroup.h"
#include "sbfGroupFilter.h"
#include "sbfPropertiesSet.h"
#include "sbfStiffMatrixBlock.h"
#include <memory>
#include <tuple>

auto makeTestData_v01 = []() {
    float xDim = 1.0;
    int xPart = 2000;
    std::unique_ptr<sbfMesh> mesh_up(std::make_unique<sbfMesh>());
    mesh_up.reset(sbfMesh::makeBlock(xDim, 0.05, 0.01, xPart, 500, 10));
    sbfMesh *mesh = mesh_up.get();

    NodesData<> displ("displ", mesh), force("force", mesh);
    force.null(); displ.null();

    sbfGroupFilter filtLock, filtLoad;
    filtLock.setCrdXF(mesh->minX()-xDim/xPart/4, mesh->minX()+xDim/xPart/4);
    filtLoad.setCrdXF(mesh->maxX()-xDim/xPart/4, mesh->maxX()+xDim/xPart/4);

    mesh->addNodeGroup(filtLock);
    mesh->addNodeGroup(filtLoad);
    mesh->processNodeGroups();

    auto listLock = mesh->nodeGroup(0)->nodeIndList();
    auto listLoad = mesh->nodeGroup(1)->nodeIndList();

    std::unique_ptr<sbfPropertiesSet> props_up(std::make_unique<sbfPropertiesSet>());
    sbfPropertiesSet *props = props_up.get();
    props->addMaterial(sbfMaterialProperties::makeMPropertiesSteel());

    report("Constructing stiff");
    std::unique_ptr<sbfStiffMatrixBlock<3>> stiff_up(std::make_unique<sbfStiffMatrixBlock<3>>(mesh, props, MatrixType::FULL_MATRIX));
    sbfStiffMatrixBlock<3> *stiff = stiff_up.get();

    report("Computing stiff");
    stiff->compute();

    //TODO test with dof locking
//    for(auto i : listLock) stiff->lockDof(i, 0, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
//    stiff->lockDof(mesh->nodeAt(0, 0, 0), 1, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
//    stiff->lockDof(mesh->nodeAt(0, 0, 0), 2, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
//    stiff->lockDof(mesh->nodeAt(0, mesh->maxY(), 0), 2, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);
//    stiff->lockDof(mesh->nodeAt(0, 0, mesh->maxZ()), 1, 0, force.data(), LockType::APPROXIMATE_LOCK_TYPE);

    mesh_up.release();
    props_up.release();
    return std::make_tuple(std::move(stiff_up), std::move(displ), std::move(force), std::move(listLock), std::move(listLoad));
};

#endif // COMMON_HPP

