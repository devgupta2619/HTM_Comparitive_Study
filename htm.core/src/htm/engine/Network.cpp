/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2017, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
Implementation of the Network class
*/

#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <htm/engine/Input.hpp>
#include <htm/engine/Link.hpp>
#include <htm/engine/Network.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/RegionImplFactory.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/os/Directory.hpp>
#include <htm/os/Path.hpp>
#include <htm/ntypes/BasicType.hpp>
#include <htm/utils/Log.hpp>
#include <htm/ntypes/Value.hpp>

namespace htm {

class RegisteredRegionImpl;

thread_local LogLevel NTA_LOG_LEVEL; 


Network::Network() {
  commonInit();
}

// move constructor
Network::Network(Network &&n) noexcept {
  regions_ = std::move(n.regions_);
  minEnabledPhase_ = n.minEnabledPhase_;
  maxEnabledPhase_ = n.maxEnabledPhase_;
  phaseInfo_ = std::move(n.phaseInfo_);
  callbacks_ = n.callbacks_;
  iteration_ = n.iteration_;
}

Network::Network(const std::string& filename) {
  commonInit();
  loadFromFile(filename);
}


void Network::commonInit() {
  initialized_ = false;
  iteration_ = 0;
  minEnabledPhase_ = 0;
  maxEnabledPhase_ = 0;
}

Network::~Network() {
  /**
   * Teardown choreography:
   * - unintialize all regions because otherwise we won't be able to disconnect
   * - remove all links, because we can't delete connected regions
   *   This also removes Input and Output objects.
   * - delete the regions themselves.
   */

  // 1. uninitialize
  for(auto p: regions_) {
    std::shared_ptr<Region> r = p.second;
    r->uninitialize();
  }

  // 2. remove all links
  for(auto p: regions_) {
    std::shared_ptr<Region> r = p.second;
    r->removeAllIncomingLinks();
  }

  // 3. delete the regions
  // They are in a map of Shared_ptr so regions are deleted when it goes out of scope.
}


void Network::configure(const std::string &yaml) {
  ValueMap vm;
  vm.parse(yaml);

  NTA_CHECK(vm.isMap() && vm.contains("network")) << "Expected yaml string to start with 'network:'.";
  Value &v1 = vm["network"];
  NTA_CHECK(v1.isSequence()) << "Expected a sequence of entries starting with a command.";
  for (size_t i = 0; i < v1.size(); i++) {
    NTA_CHECK(v1[i].isMap()) << "Expcted a command";
    for (auto cmd : v1[i]) {
      if (cmd.first == "registerRegion") {
        std::string type = cmd.second["type"].str();
        std::string path = cmd.second["path"].str();
        std::string classname = cmd.second["classname"].str();
        // TODO:  
        NTA_THROW << "For now you can only use the built-in C++ regions with the REST API.";
      } else if (cmd.first == "addRegion") {
        std::string name = cmd.second["name"].str();
        std::string type = cmd.second["type"].str();
        ValueMap params;
        if (cmd.second.contains("params")) params = cmd.second["params"];
        // The phase can be specified as part of an addRegion
        std::set<UInt32> phases;
        if (cmd.second.contains("phase")) {
          if (cmd.second["phase"].isSequence()) {
            const std::vector<int> lst = cmd.second["phase"].asVector<int>();
            for (int phase : lst) {
              NTA_CHECK(phase >= 0) << "A phase ID must be a positive integer.";
              phases.insert((UInt32)phase);
            }
          } else {
            int phase = cmd.second["phase"].as<int>();
            NTA_CHECK(phase >= 0) << "A phase ID must be a positive integer";
            phases.insert((UInt32)phase);
          }
        }
        addRegion(name, type, params, phases);

      } else if (cmd.first == "addLink") {
        // The an addLink section can specify src, dest, delay, dim, and mode
        std::string src = cmd.second["src"].str();
        std::string dest = cmd.second["dest"].str();
        std::string dim = "";
        std::string mode = "";
        std::string linkParam = "";
        std::vector<std::string> vsrc = Path::split(src, '.');
        std::vector<std::string> vdest = Path::split(dest, '.');
        int propagationDelay = 0;
        if (cmd.second.contains("delay"))
          propagationDelay = cmd.second["delay"].as<int>();
        if (cmd.second.contains("dim")) {
          dim = cmd.second["dim"].to_json();
        }
        if (cmd.second.contains("mode")) {
          mode = cmd.second["mode"].str();
        }
        if (!dim.empty() || !mode.empty()) {
          linkParam = "{";
          if (!dim.empty())
            linkParam += "dim: " + dim;
          if (!mode.empty()) {
            if (linkParam.size() > 1)
              linkParam += ", ";
            linkParam += "mode : " + mode;
          }
          linkParam += "}";
        }
        NTA_CHECK(vsrc.size() == 2) << "Expecting source domain name '.' output name.";
        NTA_CHECK(vdest.size() == 2) << "Expecting destination domain name '.' input name.";

        link(vsrc[0], vdest[0], "", linkParam, vsrc[1], vdest[1], propagationDelay);
      }
    }
  }
}

// An overload of addRegion() that can specify phases
std::shared_ptr<Region> Network::addRegion( const std::string &name, 
                                            const std::string &nodeType,
                                            const std::string &nodeParams,
                                            const std::set<UInt32> &phases) {
  if (regions_.find(name) != regions_.end())
    NTA_THROW << "Region with name '" << name << "' already exists in network";
  std::shared_ptr<Region> r = std::make_shared<Region>(name, nodeType, nodeParams, this);
  regions_[name] = r;
  initialized_ = false;

  if (phases.empty()) {
    std::set<UInt32> defaultPhase;
    defaultPhase.insert(0); // if no phase is specified, use phase 0.
    setPhases_(r, defaultPhase);
  } else
    setPhases_(r, phases);
  return r;
}

// An overload of addRegion() that defaults to phase 0
std::shared_ptr<Region> Network::addRegion(const std::string &name, 
                                           const std::string &nodeType,
                                           const std::string &nodeParams) {

  std::set<UInt32> defaultPhase;
  defaultPhase.insert(0u);
  std::shared_ptr<Region> r = addRegion(name, nodeType, nodeParams, defaultPhase);
  return r;
}

// An overload of addRegion() that allows phases and has nodeParams already parsed.
std::shared_ptr<Region> Network::addRegion(const std::string &name, 
                                           const std::string &nodeType,
                                           ValueMap& vm,
                                           const std::set<UInt32> &phases) {
  if (regions_.find(name) != regions_.end())
    NTA_THROW << "Region with name '" << name << "' already exists in network";
  std::shared_ptr<Region> r = std::make_shared<Region>(name, nodeType, vm, this);
  regions_[name] = r;
  initialized_ = false;

  if (phases.empty()) {
    std::set<UInt32> defaultPhase;
    defaultPhase.insert(0);  // if no phase is specified, use phase 0.
    setPhases_(r, defaultPhase);
  } else
    setPhases_(r, phases);
  return r;
}

// An overload of addRegon() that is used during a load to restore a region.
std::shared_ptr<Region> Network::addRegion(std::shared_ptr<Region>& r) {
  NTA_CHECK(r != nullptr);
  r->network_ = this;
  regions_[r->getName()] = r;
  
  // Note: phases are added in this case by calling phasesFromString()

  return r;
}


void Network::setPhases_(std::shared_ptr<Region> &r, const std::set<UInt32> &phases) {
  // The phases are an ordered list of phase ID's. A phase is a list
  // of regions that are executed together in the order that they are assigned to a phase
  // (which is normally when the region is created).
  //
  // This function will assign a region to a phase or set of phases.  
  // All regions not explicitly assigned to a phase are placed in phase 0.
  // The phase ID is also the index of that phase in the phaseInfo structure.  
  //
  // Normally an application that is using phases will start with phase 1 and increment
  // the phase ID by one for each additional phase they are using.
  //
  // Note: if a region is already in any phases, it will be removed from all phases
  //       which are not in the new list. It will keep its position in the 
  //       execution order for phases that already existed and are also in the new phase list.
  //
  if (phases.empty())
    NTA_THROW << "Attempt to set empty phase list for region " << r->getName();

  // Make room for more phases if needed.
  UInt32 maxNewPhase = *(phases.rbegin());  // get the last phase ID (the largest index)
  UInt32 nextPhase = (UInt32)phaseInfo_.size();
  if (maxNewPhase >= nextPhase) {
    phaseInfo_.resize((size_t)maxNewPhase + 1); 
  }
  for (UInt i = 0; i < phaseInfo_.size(); i++) {
    bool insertPhase = false;
    if (phases.find(i) != phases.end())
      insertPhase = true;  // This phase is in the new list.

    // remove previous settings for this region if no longer in a phase
    for (auto item = phaseInfo_[i].begin(); item != phaseInfo_[i].end(); item++) {
      if ((*item)->getName() == r->getName()) {
        // already in this phase.
        if (insertPhase) {
          // let the region keeps its current location in the execution sequence for this phase.
          insertPhase = false;
          break;
        } else {
          // no longer needed in this phase, remove.
          phaseInfo_[i].erase(item);
          break;
        }
      }
    }
    if (insertPhase) {
        // add the region to this phase.
      if (r->getName() == "INPUT")
        phaseInfo_[i].insert(phaseInfo_[i].begin(), r);  // INPUT region must be first in the phase
      else
        phaseInfo_[i].push_back(r);
    }
  }


  resetEnabledPhases_();
}

void Network::resetEnabledPhases_() {
  // min/max enabled phases based on what is in the network
  minEnabledPhase_ = getMinPhase();
  maxEnabledPhase_ = getMaxPhase();
}

void Network::setPhases(const std::string &name, const std::set<UInt32> &phases) {
  auto itr = regions_.find(name);
  if (itr == regions_.end())
    NTA_THROW << "setPhases -- no region exists with name '" << name << "'";
  std::shared_ptr<Region> &r = itr->second;

  if (phases.empty()) {
    std::set<UInt32> ph;
    ph.insert(0); 
    setPhases_(r, ph);
  } else
    setPhases_(r, phases);
}

std::set<UInt32> Network::getPhases(const std::string &name) const {
  auto itr = regions_.find(name);
  if (itr == regions_.end())
    NTA_THROW << "getPhases -- no region exists with name '" << name << "'";

  const std::shared_ptr<Region> r = itr->second;

  std::set<UInt32> phases;
  // construct the set of phases enabled for this region
  for (UInt32 i = 0; i < phaseInfo_.size(); i++) {
    for (auto r : phaseInfo_[i]) {
      if (r->getName() == name)
        phases.insert(i);
    }
  }
  return phases;
}

void Network::removeRegion(const std::string &name) {
  auto itr = regions_.find(name);
  if (itr == regions_.end())
    NTA_THROW << "removeRegion: no region named '" << name << "'";

  const std::shared_ptr<Region>& r = itr->second;
  if (r->hasOutgoingLinks())
    NTA_THROW << "Unable to remove region '" << name
              << "' because it has one or more outgoing links";

  // Network does not have to be uninitialized -- removing a region
  // has no effect on the network as long as it has no outgoing links,
  // which we have already checked.

  // Must uninitialize the region prior to removing incoming links
  // The incoming links are removed when the Input object is deleted.
  r->uninitialize();
  r->clearInputs();

  // take it out of all phases
  for (UInt i = 0; i < phaseInfo_.size(); i++) {
    for (auto it = phaseInfo_[i].begin(); it != phaseInfo_[i].end(); ++it) {
      if (name == (*it)->getName()) {
        phaseInfo_[i].erase(it);
        break; // should never be more than one with that name.
      }
    }
  }

  // Trim phaseinfo as we may have no more regions at the highest phase(s)
  for (size_t i = phaseInfo_.size() - 1; i > 0; i--) {
    if (phaseInfo_[i].empty())
      phaseInfo_.resize(i);
    else
      break;
  }
  resetEnabledPhases_();

  // Region is deleted when the Shared_ptr goes out of scope.
  regions_.erase(itr);
  return;
}

std::shared_ptr<Link> Network::link(const std::string &srcRegionName,
                   const std::string &destRegionName,
                   const std::string &linkType, const std::string &linkParams,
                   const std::string &srcOutputName,
                   const std::string &destInputName,
                   const size_t propagationDelay) {

  // Find the regions
  auto itrSrc = regions_.find(srcRegionName);
  if (itrSrc == regions_.end()) {
    if (srcRegionName != "INPUT")
      NTA_THROW << "Network::link -- source region '" << srcRegionName << "' does not exist";
    // Our special InputRegion does not exist. But we are using it so we need to create it.
    ValueMap vm;  // Empty parameters.
    std::set<UInt32> phases; 
    phases.insert(0u);  // (This special region goes into phase 0)
    Network::addRegion(srcRegionName, "RawInput", vm, phases);  // Note: will have no Outputs defined.
  }
  std::shared_ptr<Region> srcRegion = regions_[srcRegionName];

  auto itrDest = regions_.find(destRegionName);
  if (itrDest == regions_.end())
    NTA_THROW << "Network::link -- dest region '" << destRegionName
              << "' does not exist";
  std::shared_ptr<Region> destRegion = regions_[destRegionName];

  // Find the inputs/outputs
  // get the destination Input object
  std::string inputName = destInputName;
  if (inputName == "") {
    const std::shared_ptr<Spec> &destSpec = destRegion->getSpec();
    inputName = destSpec->getDefaultInputName();
  }

  std::shared_ptr<Input> destInput = destRegion->getInput(inputName);
  if (destInput == nullptr) {
    NTA_THROW << "Network::link -- input '" << inputName << " does not exist on region " << destRegionName;
  }

  // get the source Output object
  std::string outputName = srcOutputName;
  if (outputName == "") {
    const std::shared_ptr<Spec>& srcSpec = srcRegion->getSpec();
    outputName = srcSpec->getDefaultOutputName();
  }

  Dimensions dim;
  std::string params = Path::trim(linkParams);
  if (!params.empty()) {
    Value v;
    v.parse(linkParams);
    if (v.isMap() && v.contains("dim")) {
      Value v1 = v["dim"];
      if (v1.isSequence())
        dim = Dimensions(v1.asVector<UInt>());
      else if (v1.isScalar())
        dim = Dimensions(v1.as<UInt>());
    }
  }

  std::shared_ptr<Output> srcOutput = srcRegion->getOutput(outputName);
  if (srcOutput == nullptr) {
    if (srcRegionName == "INPUT") {
      // This is our special source region for manually setting inputs using a link
      // Get the data type from the destination Input object and create an output.
      NTA_BasicType type = destInput->getDataType();
      srcOutput = std::make_shared<Output>(srcRegion.get(), outputName, type);

      if (dim.empty())
        NTA_THROW << "Link declared with Special \"INPUT\" source requires dimensions in link parameters. Something like "
                     "{dim: 100} or {dim: [10,20]}";

      srcOutput->setDimensions(dim);
      srcOutput->initialize();  // allocates the buffer
      srcRegion->outputs_[outputName] = srcOutput;
    } else {
      NTA_THROW << "Network::link -- output " << outputName << " does not exist on region " << srcRegionName;
    }
  }

  // Create the link itself
  auto link = std::make_shared<Link>(linkType, linkParams, srcOutput, destInput, propagationDelay);
  destInput->addLink(link, srcOutput);
  return link;
}

void Network::removeLink(const std::string &srcRegionName,
                         const std::string &destRegionName,
                         const std::string &srcOutputName,
                         const std::string &destInputName) {
  // Find the regions
  auto itrSrc = regions_.find(srcRegionName);
  if (itrSrc == regions_.end())
    NTA_THROW << "Network::unlink -- source region '" << srcRegionName
              << "' does not exist";
  std::shared_ptr<Region> srcRegion = getRegion(srcRegionName);

  auto itrDest = regions_.find(destRegionName);
  if (itrDest == regions_.end())
    NTA_THROW << "Network::unlink -- dest region '" << destRegionName
              << "' does not exist";
  std::shared_ptr<Region> destRegion = getRegion(destRegionName);

  // Find the inputs
  const std::shared_ptr<Spec>& srcSpec = srcRegion->getSpec();
  const std::shared_ptr<Spec>& destSpec = destRegion->getSpec();
  std::string inputName;
  if (destInputName == "")
    inputName = destSpec->getDefaultInputName();
  else
    inputName = destInputName;

  std::shared_ptr<Input> destInput = destRegion->getInput(inputName);
  if (destInput == nullptr) {
    NTA_THROW << "Network::unlink -- input '" << inputName
              << " does not exist on region " << destRegionName;
  }

  std::string outputName = srcOutputName;
  if (outputName == "")
    outputName = srcSpec->getDefaultOutputName();
  std::shared_ptr<Link> link = destInput->findLink(srcRegionName, outputName);

  if (!link)
    NTA_THROW << "Network::unlink -- no link exists from region "
              << srcRegionName << " output " << outputName << " to region "
              << destRegionName << " input " << destInput->getName();

  // Finally, remove the link
  destInput->removeLink(link);
}

void Network::initialize() {

  /*
   * Do not reinitialize if already initialized.
   * Mostly, this is harmless, but it has a side
   * effect of resetting the max/min enabled phases,
   * which causes havoc if we are in the middle of
   * a computation.
   */
  if (initialized_)
    return;

  /*
   * 1. Calculate all Input/Output dimensions by evaluating links.
   */
  // evaluate regions and initialize their links in region execution order
  //NTA_DEBUG << "Initialize Links: " << std::endl;
  for (UInt32 phaseID = 0; phaseID < phaseInfo_.size(); phaseID++) {
    for (auto r : phaseInfo_[phaseID]) {
      //NTA_DEBUG << "  phase " << phaseID << " region: " << r->getName() << std::endl;
      r->evaluateLinks();
    }
  }

  /*
   * 2. initialize region/impl
   */
  for (auto p : regions_) {
    std::shared_ptr<Region> r = p.second;
    r->initialize();
  }

  /*
   * 3. Enable all phases in the network
   */
  resetEnabledPhases_();

  /*
   * Mark network as initialized.
   */
  initialized_ = true;
}




/**
 * Set the source data for a Link identified with the source as "INPUT" and <sourceName>.
 * NOTE: we need to do the INPUT link because the archive stores only outputs.  So the initial
 *       input needs an output to be stored on.  Otherwise, data written directly to input buffers 
 *       would be lost during a load() from archive.
 */
void Network::setInputData(const std::string& sourceName, const Array& data) {
  if (!initialized_) {
    initialize();
  }
  // The placeholder region "INPUT" with an output of <sourceName> should already exist if the link was defined.
  std::shared_ptr<Region> region = getRegion("INPUT"); 
  Array &a = region->getOutput(sourceName)->getData(); // we actually populate an output buffer that will be moved to input.
  NTA_CHECK(a.getCount() == data.getCount())
      << "setInputData: Number of elements in buffer ( " << a.getCount() << " ) do not match target dimensions.";
  if (a.getType() == data.getType()) {
    a = data;  // assign the buffer without copy
  }  else {
    data.convertInto(a);  // copy the data with conversion.
  }
}

void Network::setInputData(const std::string &sourceName, const Value& vm) {
  if (!initialized_) {
    initialize();
  }
  // The placeholder region "INPUT" with an output of <sourceName> should already exist if the link was defined.
  std::shared_ptr<Region> region = getRegion("INPUT");
  Array &a =  region->getOutput(sourceName)->getData(); // populate this output buffer that will be moved to the input.
  NTA_BasicType type = a.getType();

  NTA_CHECK(vm.contains("data"))
      << "Unexpected YAML or JSON format. Expecting something like {data: [1,0,1]}";

  NTA_CHECK(vm["data"].isSequence())
      << "Unexpected YAML or JSON format. Expecting something like {data: [1,0,1]}";

  if (type == NTA_BasicType_SDR) {
    NTA_CHECK(a.getCount() >= vm.size())
        << "setInputData: Number of elements in buffer ( " << a.getCount() << " ) do not match target dimensions.";
  } else {
    NTA_CHECK(a.getCount() == vm.size())
        << "setInputData: Number of elements in buffer ( " << a.getCount() << " ) do not match target dimensions.";
  }

  a.fromValue(vm);
}


void Network::run(int n, std::vector<UInt32> phases) {
  if (!initialized_) {
    initialize();
  }

  if (phaseInfo_.empty())
    return;

  NTA_CHECK(maxEnabledPhase_ < phaseInfo_.size())
      << "maxphase: " << maxEnabledPhase_ << " size: " << phaseInfo_.size();


  for (int iter = 0; iter < n; iter++) {
    iteration_++;

    // In phase order and order of region definition within a phase, execute each region.
    // After executing a region, move the output buffer to the input buffer for each link.
    if (!phases.empty()) {
      // execute the specified phases only.
      for (UInt32 phase : phases) {
        NTA_CHECK(phase < phaseInfo_.size()) << "Phase ID " << phase << " specified in run() is out of range.";
        for (auto r : phaseInfo_[phase]) {
          //r->prepareInputs();       // For each link pull sources for each input
          r->compute();
          r->pushOutputsOverLinks();  // copy outputs to inputs for each link.
        }
      }
    } else {
      // compute all enabled regions in phase order, within a phase
      // execute regions in the order they were placed into the phase.
      for (UInt32 current_phase = minEnabledPhase_; current_phase <= maxEnabledPhase_; current_phase++) {
        for (auto r : phaseInfo_[current_phase]) {
          //r->prepareInputs();       // For each link pull sources for each input
          r->compute();
          r->pushOutputsOverLinks();  // copy outputs to inputs for each link.
        }
      }
    }

    // invoke callbacks
    for (UInt32 i = 0; i < callbacks_.getCount(); i++) {
      const std::pair<std::string, callbackItem> &callback = callbacks_.getByIndex(i);
      callback.second.first(this, iteration_, callback.second.second);
    }

    /**** we are now doing the delay buffer shift when output is being distributed.
     *
     *    // Refresh all links in the network at the end of every timestamp so that
     *    // data in delayed links appears to change atomically between iterations
     *    for (auto p: regions_) {
     *      const std::shared_ptr<Region> r = p.second;
     *
     *      for (const auto &inputTuple : r->getInputs()) {
     *        for (const auto &pLink : inputTuple.second->getLinks()) {
     *          pLink->shiftBufferedData();
     *        }
     *      }
     *    }
     ******/

  } // End of outer run-loop

  return;
}



const Collection<std::shared_ptr<Region>> Network::getRegions() const { 
  Collection<std::shared_ptr<Region>> regions;
  for(auto r: regions_) {
    regions.add(r.first, r.second);
  }
  return regions; 
}

std::shared_ptr<Region> Network::getRegion(const std::string& name) const {
  auto itr = regions_.find(name);
  if (itr == regions_.end())
    NTA_THROW << "Network::getRegion; '" << name << "' does not exist";
  return itr->second;
}

std::string Network::getSpecJSON(const std::string &type) {
  RegionImplFactory &factory = RegionImplFactory::getInstance();
  const std::shared_ptr<Spec> spec = factory.getSpec(type);
  return spec->toString();
}



std::vector<std::shared_ptr<Link>> Network::getLinks() const {
  std::vector<std::shared_ptr<Link>> links;

  for (auto p : regions_) {
    for (auto &input : p.second->getInputs()) {
      for (auto &link : input.second->getLinks()) {
        links.push_back(link);
      }
    }
  }

  return links;
}

Collection<Network::callbackItem> &Network::getCallbacks() {
  return callbacks_;
}

UInt32 Network::getMinPhase() const {
  UInt32 i = 0;
  for (; i < phaseInfo_.size(); i++) {
    if (!phaseInfo_[i].empty())
      break;
  }
  return i;
}

UInt32 Network::getMaxPhase() const {
  /*
   * phaseInfo_ is always trimmed, so the max phase is
   * phaseInfo_.size()-1
   */

  if (phaseInfo_.empty())
    return 0;

  return (UInt32)(phaseInfo_.size() - 1);
}

void Network::setMinEnabledPhase(UInt32 minPhase) {
  if (minPhase >= phaseInfo_.size())
    NTA_THROW << "Attempt to set min enabled phase " << minPhase
              << " which is larger than the highest phase in the network - "
              << phaseInfo_.size() - 1;
  minEnabledPhase_ = minPhase;
}

void Network::setMaxEnabledPhase(UInt32 maxPhase) {
  if (maxPhase >= phaseInfo_.size())
    NTA_THROW << "Attempt to set max enabled phase " << maxPhase
              << " which is larger than the highest phase in the network - "
              << phaseInfo_.size() - 1;
  maxEnabledPhase_ = maxPhase;
}

UInt32 Network::getMinEnabledPhase() const { return minEnabledPhase_; }

UInt32 Network::getMaxEnabledPhase() const { return maxEnabledPhase_; }



void Network::post_load(std::vector<std::shared_ptr<Link>>& links) {
    for(auto alink: links) {
      auto l = link( alink->getSrcRegionName(),
                     alink->getDestRegionName(),
                     "", "",
                     alink->getSrcOutputName(),
                     alink->getDestInputName(),
                     alink->getPropagationDelay());
      l->propagationDelayBuffer_ = alink->propagationDelayBuffer_;
    }
    post_load();
}

void Network::post_load() {
  // Post Load operations
  for(auto p: regions_) {
    std::shared_ptr<Region>& r = p.second;
    r->network_ = this;
    r->evaluateLinks();      // Create the input buffers.
  }


  // Note: When serialized, the output buffers are saved
  //       by each Region.  After restore we need to
  //       copy restored outputs to connected inputs.
  //
  //       Input buffers are not saved, they are restored by
  //       copying from their source output buffers via links.
  //       If an input is manually set then the input would be
  //       lost after restore.
  
  for (auto p: regions_) {
    std::shared_ptr<Region>&  r = p.second;

    // If a propogation Delay is specified, the Link serialization
	  // saves the current input buffer at the top of the
	  // propogation Delay array because it will be pushed to
	  // the input during run();
	  // It then saves all but the back buffer
    // (the most recent) of the Propogation Delay array because
    // that buffer is the same as the most current output.
	  // So after restore we need to call prepareInputs() and
	  // shift the current outputs into the Propogaton Delay array.

    r->prepareInputs();  // for each link, pull sources for each input.
                         // it will also shift the Propogaton Delay data.

  }

  // If we made it this far without an exception, we are good to go.
  initialized_ = true;

}

/**
 * A simple function to display the execution map. Show what regions are in which phase
 * how the dimensions were propogated.  Intended for debugging an application.
 */
std::string Network::getExecutionMap() {
  std::stringstream ss;
  ss << "   Execution Map \n";

  for (UInt32 phaseID = 0; phaseID < phaseInfo_.size(); phaseID++) {
    if (!phaseInfo_[phaseID].empty()) {
      ss << "  Phase" << phaseID << ((phaseID < minEnabledPhase_ || phaseID > maxEnabledPhase_) ? " (disabled)" : "") << std::endl;
      for (auto r : phaseInfo_[phaseID]) {
        ss << "    region: " << r->getName() << std::endl;
        auto outputs = r->getOutputs();
        for (auto it = outputs.begin(); it != outputs.end(); it++) {
          const std::set<std::shared_ptr<Link>> &links = it->second->getLinks();
          for (auto link : links) {
            ss << "      " << link->toString() << std::endl;
          }
        }
      }
    }
  }
  return ss.str();
}

std::string Network::phasesToString() const {
  std::stringstream ss;
  ss << "{";
  ss << "minEnabledPhase_: " << minEnabledPhase_ << ", ";
  ss << "maxEnabledPhase_: " << maxEnabledPhase_ << ", ";
  ss << "phases: [";
  for (auto phase : phaseInfo_) {
    ss << "[";
    for (auto region : phase) {
      ss << region->getName() << ", ";
    }
    ss << "]";
  }
  ss << "]}";
  return ss.str();
}
void Network::phasesFromString(const std::string& phaseString) {
  std::string content = phaseString;
  content.erase(std::remove(content.begin(), content.end(), ','), content.end());
  std::stringstream ss(content);
  std::string tag;
  std::vector<std::shared_ptr<Region>> phase;
  
  NTA_CHECK(ss.peek() == '{') << "Invalid phase deserialization";
  ss.ignore(1);
  ss >> tag;
  NTA_CHECK(tag == "minEnabledPhase_:");
  ss >> minEnabledPhase_;
  ss >> tag;
  NTA_CHECK(tag == "maxEnabledPhase_:");
  ss >> maxEnabledPhase_;
  ss >> tag;
  NTA_CHECK(tag == "phases:") << "Invalid phase deserialization";
  ss >> std::ws;
  NTA_CHECK(ss.peek() == '[') << "Invalid phase deserialization";
  ss.ignore(1);
  ss >> std::ws;
  while (ss.peek() != ']') {
    ss >> std::ws;
    if (ss.peek() == '[') {
      ss.ignore(1);
      ss >> std::ws;
      while (ss.peek() != ']') {
        ss >> tag;
        auto it = regions_.find(tag);
        NTA_CHECK(it != regions_.end()) << "Region '" << tag << "' not found while decoding phase.";
        phase.push_back(it->second);
        ss >> std::ws;
      }
      ss.ignore(1); // ']'
      phaseInfo_.push_back(phase);
      phase.clear();
    }
  }
  ss >> std::ws;
  ss.ignore(1); // ']'
}


void Network::enableProfiling() {
  for (auto p: regions_) {
    std::shared_ptr<Region> r = p.second;
    r->enableProfiling();
  }
}

void Network::disableProfiling() {
  for (auto p: regions_) {
    std::shared_ptr<Region> r = p.second;
    r->disableProfiling();
  }
}

void Network::resetProfiling() {
  for (auto p: regions_) {
    std::shared_ptr<Region>  r = p.second;
    r->resetProfiling();
  }
}

  /*
   * Adds a region to the RegionImplFactory's list of packages
   */
void Network::registerRegion(const std::string name, RegisteredRegionImpl *wrapper) {
	RegionImplFactory::registerRegion(name, wrapper);
}
  /*
   * Removes a region from RegionImplFactory's list of packages
   */
void Network::unregisterRegion(const std::string name) {
	RegionImplFactory::unregisterRegion(name);
}

std::string Network::getRegistrations() {
  return RegionImplFactory::getRegistrations();
}

void Network::cleanup() {
    RegionImplFactory::cleanup();
}

bool Network::operator==(const Network &o) const {

  if (initialized_ != o.initialized_ || iteration_ != o.iteration_ ||
      minEnabledPhase_ != o.minEnabledPhase_ ||
      maxEnabledPhase_ != o.maxEnabledPhase_ ||
      regions_.size() != o.regions_.size()) {
    return false;
  }

  for(auto iter = regions_.cbegin(); iter != regions_.cend(); ++iter){
    std::shared_ptr<Region> r1 = iter->second;
    std::string name = r1->getName();
    auto itr = o.regions_.find(name);
    if (itr == o.regions_.end()) return false;
    std::shared_ptr<Region> r2 = itr->second;
    if (*(r1.get()) != *(r2.get())) {
      return false;
    }
  }

  if (phaseInfo_.size() != o.phaseInfo_.size())
    return false;
  for (size_t phase = 0; phase < phaseInfo_.size(); phase++) {
    if (phaseInfo_[phase].size() != o.phaseInfo_[phase].size())
      return false;
    for (size_t i = 0; i < phaseInfo_[phase].size(); i++) {
      if (phaseInfo_[phase][i]->getName() != o.phaseInfo_[phase][i]->getName())
        return false;
    }
  }
  return true;
}

std::ostream &operator<<(std::ostream &f, const Network &n) {
  // Display Network, Region, Links

  f << "Network: {\n";
  f << "iteration: " << n.iteration_ << "\n";
  f << "Regions: " << "[\n";

  for(auto iter = n.regions_.cbegin(); iter != n.regions_.cend(); ++iter) {
      std::shared_ptr<Region>  r = iter->second;
      f << (*r.get());
  }
  f << "]\n"; // end of regions

  // Display the Links
  f << "Links: [\n";
  for(auto iter = n.regions_.cbegin(); iter != n.regions_.cend(); ++iter) {
    std::shared_ptr<Region> r = iter->second;
    const std::map<std::string, std::shared_ptr<Input>> inputs = r->getInputs();
    for (const auto & inputs_input : inputs)
    {
      const std::vector<std::shared_ptr<Link>>& links = inputs_input.second->getLinks();
      for (const auto & links_link : links)
      {
        auto l = links_link;
        f << (*l.get());
      }

    }
  }
  f << "]\n"; // end of links

  f << "}\n"; // end of network
  f << std::endl;
  return f;
}



} // namespace htm
