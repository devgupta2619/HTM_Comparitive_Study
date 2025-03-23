/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * Implementation of Network test
 */

#include "gtest/gtest.h"

#include <htm/engine/Network.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/ntypes/Dimensions.hpp>
#include <htm/engine/RegionImpl.hpp>
#include <htm/engine/RegisteredRegionImplCpp.hpp>
#include <htm/utils/Log.hpp>

namespace testing {

using namespace htm;

static bool verbose = false;
#define VERBOSE if(verbose) std::cerr << "[          ]"

TEST(NetworkTest, RegionAccess) {
  Network net;
  EXPECT_THROW(net.addRegion("level1", "nonexistent_nodetype", ""), std::exception);

  // Should be able to add a region
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");

  ASSERT_TRUE(l1->getNetwork() == &net);

  EXPECT_THROW(net.getRegion("nosuchregion"), std::exception);

  // Make sure partial matches don't work
  EXPECT_THROW(net.getRegion("level"), std::exception);

  std::shared_ptr<Region> l1a = net.getRegion("level1");
  ASSERT_TRUE(l1a == l1);

  // Should not be able to add a second region with the same name
  EXPECT_THROW(net.addRegion("level1", "TestNode", ""), std::exception);
}

TEST(NetworkTest, InitializationBasic) {
  Network net;
  net.initialize();
}

TEST(NetworkTest, InitializationNoRegions) {
  Network net;
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");

  // Region does not yet have dimensions -- prevents network initialization

  Dimensions d;
  d.push_back(4);
  d.push_back(4);

  l1->setDimensions(d);

  // Should succeed since dimensions are now set
  net.initialize();
  net.run(1);

  std::shared_ptr<Region> l2 = net.addRegion("level2", "TestNode", "");
  l2->setDimensions(d);
  net.run(1);
}

TEST(NetworkTest, Modification) {
  NTA_DEBUG << "Running network modification tests";

  Network net;
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");

  // should have been added at phase0
  std::set<UInt32> phases = net.getPhases("level1");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(0) != phases.end());

  Dimensions d;
  d.push_back(4);
  d.push_back(4);
  l1->setDimensions(d);

  std::shared_ptr<Region> l2 = net.addRegion("level2", "TestNode", "{dim: [2,2]}");

  // should have been added into phase0
  phases = net.getPhases("level2");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(0) != phases.end());

  net.link("level1", "level2");

  ASSERT_EQ((UInt32)2, net.getRegions().size());

  // Should succeed since dimensions are set
  net.initialize();
  net.run(1);
  l2 = net.getRegion("level2");
  Dimensions d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)2, d2[0]);
  ASSERT_EQ((UInt32)2, d2[1]);

  EXPECT_THROW(net.removeRegion("doesntexist"), std::exception);

  net.removeRegion("level2");
  // net now only contains level1
  ASSERT_EQ((UInt32)1, net.getRegions().size()) << "Should be only region 'level1' remaining\n";
  EXPECT_THROW(net.getRegion("level2"), std::exception);

  auto links = net.getLinks();
  ASSERT_TRUE(links.size() == 0) << "Removing the destination region hould have removed the link.";

  ASSERT_TRUE(l1 == net.getRegion("level1"));
  l2 = net.addRegion("level2", "TestNode", "dim: [2,2]");

  // should have been added at phase0
  phases = net.getPhases("level2");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(0) != phases.end());

  net.link("level1", "level2");

  // network can be initialized now
  net.run(1);

  ASSERT_EQ((UInt32)2, net.getRegions().size());
  ASSERT_TRUE(l2 == net.getRegion("level2"));

  d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)2, d2[0]);
  ASSERT_EQ((UInt32)2, d2[1]);

  // add a third region
  std::shared_ptr<Region> l3 = net.addRegion("level3", "TestNode", "{dim: [1,1]}");

  // should have been added at phase 0
  phases = net.getPhases("level3");
  ASSERT_EQ((UInt32)1, phases.size());
  ASSERT_TRUE(phases.find(0) != phases.end());

  ASSERT_EQ((UInt32)3, net.getRegions().size());

  net.link("level2", "level3");
  net.initialize();
  d2 = l3->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)1, d2[0]);
  ASSERT_EQ((UInt32)1, d2[1]);

  // try to remove a region whose outputs are connected
  // this should fail because it would leave the network
  // unrunnable
  EXPECT_THROW(net.removeRegion("level2"), std::exception);
  ASSERT_EQ((UInt32)3, net.getRegions().size());
  EXPECT_THROW(net.removeRegion("level1"), std::exception);
  ASSERT_EQ((UInt32)3, net.getRegions().size());

  // this should be ok
  net.removeRegion("level3");
  ASSERT_EQ((UInt32)2, net.getRegions().size());

  net.removeRegion("level2");
  net.removeRegion("level1");
  ASSERT_EQ((UInt32)0, net.getRegions().size());

  // build up the network again -- slightly differently with
  // l1->l2 and l1->l3
  l1 = net.addRegion("level1", "TestNode", "");
  l1->setDimensions(d);
  net.addRegion("level2", "TestNode", "");
  net.addRegion("level3", "TestNode", "");
  net.link("level1", "level2");
  net.link("level1", "level3");
  net.initialize();

  // build it up one more time and let the destructor take care of it
  net.removeRegion("level2");
  net.removeRegion("level3");
  net.run(1);

  l2 = net.addRegion("level2", "TestNode", "");
  l3 = net.addRegion("level3", "TestNode", "");
  // try links in reverse order
  net.link("level2", "level3");
  net.link("level1", "level2");
  net.initialize();
  d2 = l3->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)4, d2[0]);
  ASSERT_EQ((UInt32)4, d2[1]);

  d2 = l2->getDimensions();
  ASSERT_EQ((UInt32)2, d2.size());
  ASSERT_EQ((UInt32)4, d2[0]);
  ASSERT_EQ((UInt32)4, d2[1]);

  // now let the destructor remove everything
}

TEST(NetworkTest, Unlinking) {
  VERBOSE << "Running unlinking tests \n";
  Network net;
  net.addRegion("level1", "TestNode", "");
  net.addRegion("level2", "TestNode", "");
  Dimensions d;
  d.push_back(4);
  d.push_back(2);
  net.getRegion("level1")->setDimensions(d);

  net.link("level1", "level2");
  ASSERT_TRUE(net.getRegion("level2")->getDimensions().isUnspecified());

  EXPECT_THROW(net.removeLink("level1", "level2", "outputdoesnotexist", "bottomUpIn"), std::exception);
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut", "inputdoesnotexist"), std::exception);
  EXPECT_THROW(net.removeLink("level1", "leveldoesnotexist"), std::exception);
  EXPECT_THROW(net.removeLink("leveldoesnotexist", "level2"), std::exception);

  // remove the link from the uninitialized network
  net.removeLink("level1", "level2");
  ASSERT_TRUE(net.getRegion("level2")->getDimensions().isUnspecified());

  EXPECT_THROW(net.removeLink("level1", "level2"), std::exception);

  // remove, specifying output/input names
  net.link("level1", "level2");
  net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn");
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut", "bottomUpIn"), std::exception);

  net.link("level1", "level2");
  net.removeLink("level1", "level2", "bottomUpOut");
  EXPECT_THROW(net.removeLink("level1", "level2", "bottomUpOut"), std::exception);

  // add the link back and initialize (inducing dimensions)
  net.link("level1", "level2");
  net.initialize();

  d = net.getRegion("level2")->getDimensions();
  ASSERT_EQ((UInt32)2, d.size());
  ASSERT_EQ((UInt32)4, d[0]);
  ASSERT_EQ((UInt32)2, d[1]);

  // remove the link. This will fail because we can't
  // remove a link to an initialized region
  EXPECT_THROW(net.removeLink("level1", "level2"), std::exception)
      << "Cannot remove link [level1.bottomUpOut (region dims: [4 2])  to "
         "level2.bottomUpIn (region dims: [2 1])  type: TestFanIn2] because "
         "destination region level2 is initialized. Remove the region first.";
}

typedef std::vector<std::string> callbackData;
callbackData mydata;

void testCallback(Network *net, UInt64 iteration, void *data) {
  callbackData &thedata = *(static_cast<callbackData *>(data));
  // push region names onto callback data
  const Collection<std::shared_ptr<Region>> &regions = net->getRegions();
  for (auto iter = regions.cbegin(); iter != regions.cend(); ++iter) {
    thedata.push_back(iter->first);
  }
}

std::vector<std::string> computeHistory;
static void recordCompute(const std::string &name) { computeHistory.push_back(name); }

TEST(NetworkTest, Phases) {
  Network net;

  // should auto-initialize with phase 0
  std::shared_ptr<Region> l1 = net.addRegion("level1", "TestNode", "");
  EXPECT_STREQ("level1", l1->getName().c_str());

  std::set<UInt32> phaseSet = net.getPhases("level1");
  ASSERT_EQ((UInt32)1, phaseSet.size());
  ASSERT_TRUE(phaseSet.find(0) != phaseSet.end());

  std::shared_ptr<Region> l2 = net.addRegion("level2", "TestNode", "");
  EXPECT_STREQ("level2", l2->getName().c_str());
  phaseSet = net.getPhases("level2");
  ASSERT_TRUE(phaseSet.size() == 1);
  ASSERT_TRUE(phaseSet.find(0) != phaseSet.end());

  Dimensions d;
  d.push_back(2);
  d.push_back(2);

  l1->setDimensions(d);
  l2->setDimensions(d);
  net.initialize();
  l1->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  l2->setParameterUInt64("computeCallback", (UInt64)recordCompute);

  computeHistory.clear();
  net.run(2);
  ASSERT_EQ((UInt32)4, computeHistory.size());
  // use at() to throw an exception if out of range
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());
  EXPECT_STREQ("level1", computeHistory.at(2).c_str());
  EXPECT_STREQ("level2", computeHistory.at(3).c_str());
  computeHistory.clear();

  phaseSet.clear();
  phaseSet.insert(0);
  phaseSet.insert(2);
  net.setPhases("level1", phaseSet);  // moved from phase 0 to phase0 and phase2
  net.run(2);
  ASSERT_EQ((UInt32)6, computeHistory.size());
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());
  EXPECT_STREQ("level1", computeHistory.at(2).c_str());
  EXPECT_STREQ("level1", computeHistory.at(3).c_str());
  EXPECT_STREQ("level2", computeHistory.at(4).c_str());
  EXPECT_STREQ("level1", computeHistory.at(5).c_str());
  computeHistory.clear();
}

TEST(NetworkTest, MinMaxPhase) {
  Network n;
  UInt32 minPhase = n.getMinPhase();
  UInt32 maxPhase = n.getMaxPhase();

  ASSERT_EQ((UInt32)0, minPhase);
  ASSERT_EQ((UInt32)0, maxPhase);

  EXPECT_THROW(n.setMinEnabledPhase(1), std::exception);
  EXPECT_THROW(n.setMaxEnabledPhase(1), std::exception);
  std::shared_ptr<Region> l1 = n.addRegion("level1", "TestNode", "", 1);
  std::shared_ptr<Region> l2 = n.addRegion("level2", "TestNode", "", 2);
  std::shared_ptr<Region> l3 = n.addRegion("level3", "TestNode", "", 3);
  std::shared_ptr<Region> l4 = n.addRegion("level4", "TestNode", "", {1,2});
  Dimensions d;
  d.push_back(1);
  l1->setDimensions(d);
  l2->setDimensions(d);
  l3->setDimensions(d);
  l4->setDimensions(d);

  n.initialize();

  l1->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  l2->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  l3->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  l4->setParameterUInt64("computeCallback", (UInt64)recordCompute);

  minPhase = n.getMinEnabledPhase();
  maxPhase = n.getMaxEnabledPhase();

  // everything is in phase 1, phase 2, and phase3
  //  phase 1 contains level1 and level4
  //  phase 2 contains level2 and level4
  //  phase 3 contains level3
  EXPECT_STREQ(
      "{minEnabledPhase_: 1, maxEnabledPhase_: 3, phases: [[][level1, level4, ][level2, level4, ][level3, ]]}",
      n.phasesToString().c_str());
  ASSERT_EQ((UInt32)1, minPhase);
  ASSERT_EQ((UInt32)3, maxPhase);

  computeHistory.clear();
  n.run(2);  // execute all phases
  ASSERT_EQ((UInt32)10, computeHistory.size());
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level4", computeHistory.at(1).c_str());
  EXPECT_STREQ("level2", computeHistory.at(2).c_str());
  EXPECT_STREQ("level4", computeHistory.at(3).c_str());
  EXPECT_STREQ("level3", computeHistory.at(4).c_str());
  EXPECT_STREQ("level1", computeHistory.at(5).c_str());
  EXPECT_STREQ("level4", computeHistory.at(6).c_str());
  EXPECT_STREQ("level2", computeHistory.at(7).c_str());
  EXPECT_STREQ("level4", computeHistory.at(8).c_str());
  EXPECT_STREQ("level3", computeHistory.at(9).c_str());

  // execute only phase 1 and phase 2 using the phase enable
  n.setMinEnabledPhase(1);
  n.setMaxEnabledPhase(2);
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)8, computeHistory.size());
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level4", computeHistory.at(1).c_str());
  EXPECT_STREQ("level2", computeHistory.at(2).c_str());
  EXPECT_STREQ("level4", computeHistory.at(3).c_str());
  EXPECT_STREQ("level1", computeHistory.at(4).c_str());
  EXPECT_STREQ("level4", computeHistory.at(5).c_str());
  EXPECT_STREQ("level2", computeHistory.at(6).c_str());
  EXPECT_STREQ("level4", computeHistory.at(7).c_str());

  n.setMinEnabledPhase(2);
  n.setMaxEnabledPhase(2);
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)4, computeHistory.size());
  EXPECT_STREQ("level2", computeHistory.at(0).c_str());
  EXPECT_STREQ("level4", computeHistory.at(1).c_str());
  EXPECT_STREQ("level2", computeHistory.at(2).c_str());
  EXPECT_STREQ("level4", computeHistory.at(3).c_str());

  computeHistory.clear();
  n.run(2, 3);  // Just execute phase 3
  ASSERT_EQ((UInt32)2, computeHistory.size());
  EXPECT_STREQ("level3", computeHistory.at(0).c_str());
  EXPECT_STREQ("level3", computeHistory.at(1).c_str());



  // reset to all phases enabled
  // make sure all can still be executed
  n.setMinEnabledPhase(0);
  n.setMaxEnabledPhase(n.getMaxPhase());
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)10, computeHistory.size());
  EXPECT_STREQ("level1", computeHistory.at(0).c_str());
  EXPECT_STREQ("level4", computeHistory.at(1).c_str());
  EXPECT_STREQ("level2", computeHistory.at(2).c_str());
  EXPECT_STREQ("level4", computeHistory.at(3).c_str());
  EXPECT_STREQ("level3", computeHistory.at(4).c_str());
  EXPECT_STREQ("level1", computeHistory.at(5).c_str());
  EXPECT_STREQ("level4", computeHistory.at(6).c_str());
  EXPECT_STREQ("level2", computeHistory.at(7).c_str());
  EXPECT_STREQ("level4", computeHistory.at(8).c_str());
  EXPECT_STREQ("level3", computeHistory.at(9).c_str());

  // max < min; allowed, but network should not run
  n.setMinEnabledPhase(1);
  n.setMaxEnabledPhase(0);
  computeHistory.clear();
  n.run(2);
  ASSERT_EQ((UInt32)0, computeHistory.size());

  // max > network max
  EXPECT_THROW(n.setMaxEnabledPhase(4), std::exception);


  // delete two regions
  // Place level2 in two new phases
  std::set<UInt32> phases;
  phases.insert(4);
  phases.insert(6);
  n.setPhases("level2", phases);
  n.removeRegion("level1");
  n.removeRegion("level4");
  // we now have: 
  //  phase 1 empty
  //  phase 2 empty
  //  phase 3 contains level3
  //  phase 4 contains level2
  //  phase 5 empty
  //  phase 6 contains level2
  EXPECT_STREQ(
      "{minEnabledPhase_: 3, maxEnabledPhase_: 6, phases: [[][][][level3, ][level2, ][][level2, ]]}",
      n.phasesToString().c_str());

  minPhase = n.getMinPhase();
  maxPhase = n.getMaxPhase();

  ASSERT_EQ((UInt32)3, minPhase);
  ASSERT_EQ((UInt32)6, maxPhase);

  computeHistory.clear();
  n.run(2);

  ASSERT_EQ((UInt32)6, computeHistory.size());
  EXPECT_STREQ("level3", computeHistory.at(0).c_str());
  EXPECT_STREQ("level2", computeHistory.at(1).c_str());
  EXPECT_STREQ("level2", computeHistory.at(2).c_str());
  EXPECT_STREQ("level3", computeHistory.at(3).c_str());
  EXPECT_STREQ("level2", computeHistory.at(4).c_str());
  EXPECT_STREQ("level2", computeHistory.at(5).c_str());
}

TEST(NetworkTest, Callback) {
  Network n;
  n.addRegion("level1", "TestNode", "");
  n.addRegion("level2", "TestNode", "");
  n.addRegion("level3", "TestNode", "");
  Dimensions d;
  d.push_back(1);
  n.getRegion("level1")->setDimensions(d);
  n.getRegion("level2")->setDimensions(d);
  n.getRegion("level3")->setDimensions(d);

  Collection<Network::callbackItem> &callbacks = n.getCallbacks();
  Network::callbackItem callback(testCallback, (void *)(&mydata));
  callbacks.add("Test Callback", callback);

  n.run(2);
  ASSERT_EQ((UInt32)6, mydata.size());
  EXPECT_STREQ("level1", mydata[0].c_str());
  EXPECT_STREQ("level2", mydata[1].c_str());
  EXPECT_STREQ("level3", mydata[2].c_str());
  EXPECT_STREQ("level1", mydata[3].c_str());
  EXPECT_STREQ("level2", mydata[4].c_str());
  EXPECT_STREQ("level3", mydata[5].c_str());
}

TEST(NetworkTest, Scenario1) {
  // This is intended to demonstrate execution order as A B C D C D C D E
  // Here we create three phases, phase 1 containing A and B, 
  // phase 2 containing phases C and D, and phase 3 containing E.
  Network n;

  std::shared_ptr<Region> regionA = n.addRegion("A", "TestNode", "{dim: [1]}", 1);
  std::shared_ptr<Region> regionB = n.addRegion("B", "TestNode", "{dim: [1]}", 1);
  std::shared_ptr<Region> regionC = n.addRegion("C", "TestNode", "{dim: [1]}", 2);
  std::shared_ptr<Region> regionD = n.addRegion("D", "TestNode", "{dim: [1]}", 2);
  std::shared_ptr<Region> regionE = n.addRegion("E", "TestNode", "{dim: [1]}", 3);

  regionA->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  regionB->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  regionC->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  regionD->setParameterUInt64("computeCallback", (UInt64)recordCompute);
  regionE->setParameterUInt64("computeCallback", (UInt64)recordCompute);

  computeHistory.clear();
  n.run(1, 1); // execute A, B once
  n.run(3, 2); // execute C, D three times
  n.run(1, 3); // execute E once

  ASSERT_EQ(9u, computeHistory.size());
  EXPECT_STREQ("A", computeHistory.at(0).c_str());
  EXPECT_STREQ("B", computeHistory.at(1).c_str());
  EXPECT_STREQ("C", computeHistory.at(2).c_str());
  EXPECT_STREQ("D", computeHistory.at(3).c_str());
  EXPECT_STREQ("C", computeHistory.at(4).c_str());
  EXPECT_STREQ("D", computeHistory.at(5).c_str());
  EXPECT_STREQ("C", computeHistory.at(6).c_str());
  EXPECT_STREQ("D", computeHistory.at(7).c_str());
  EXPECT_STREQ("E", computeHistory.at(8).c_str());


  n.run(1, {1, 2});   // execute just phase 1 and 2
  n.run(1);           // execute all phases
}

/**
 * Test operator '=='
 */
TEST(NetworkTest, testEqualsOperator) {
  Network n1;
  Network n2;
  ASSERT_TRUE(n1 == n2);
  Dimensions d;
  d.push_back(4);
  d.push_back(4);

  auto l1 = n1.addRegion("level1", "TestNode", "");
  ASSERT_TRUE(n1 != n2);
  auto l2 = n2.addRegion("level1", "TestNode", "");
  ASSERT_TRUE(n1 == n2);
  l1->setDimensions(d);
  ASSERT_TRUE(n1 != n2);
  l2->setDimensions(d);
  ASSERT_TRUE(n1 == n2);

  n1.addRegion("level2", "TestNode", "");
  ASSERT_TRUE(n1 != n2);
  n2.addRegion("level2", "TestNode", "");
  ASSERT_TRUE(n1 == n2);

  n1.link("level1", "level2");
  ASSERT_TRUE(n1 != n2);
  n2.link("level1", "level2");
  ASSERT_TRUE(n1 == n2);

  n1.run(1);
  ASSERT_TRUE(n1 != n2);
  n2.run(1);
  ASSERT_TRUE(n1 == n2);
}
} // namespace testing

namespace htm {
// Note: this region sort-of mimics the test in network_test.py "testNetworkPickle".
//       This will pass on an input integer array to output, incrementing the first element.
//       It also does a "HelloWorld" with the executeCommand.
class PassthruRegion : public RegionImpl {
public:
  PassthruRegion(const ValueMap &params, Region *region) : RegionImpl(region) { param = 52; }
  PassthruRegion(ArWrapper &wrapper, Region *region) : RegionImpl(region) { cereal_adapter_load(wrapper); }

  void initialize() override {}
  void compute() override {
    // This will pass its inputs on to the outputs.
    // The first element is incremented on each execution
    Array &input_data = getInput("input_u")->getData();
    ((UInt32 *)(input_data.getBuffer()))[0]++;  // increment element [0]
    Array &output_data = getOutput("output_u")->getData();
    input_data.convertInto(output_data);
  }
  // We want the dimensions to propogate so return size 0 which means "don't care".
  // The link initialization will figure out how to set it.
  size_t getNodeOutputElementCount(const std::string &name) const override { return 0; }

  std::string executeCommand(const std::vector<std::string> &args, Int64 index) override {
    if (args[0] == "HelloWorld" && args.size() == 3)
      return "Hello World says: arg1=" + args[1] + " arg2=" + args[2];
    return "";
  }

  // Include the required code for serialization.
  CerealAdapter;
  template <class Archive> void save_ar(Archive &ar) const {
    ar(cereal::make_nvp("param", param));
  }
  template <class Archive> void load_ar(Archive &ar) {
    ar(cereal::make_nvp("param", param));
  }

  static Spec *createSpec() {
    auto ns = new Spec;
    ns->description = "PassthruRegion. Used as a plain simple plugin Region for unit tests only. "
                      "This is not useful for any real applicaton. The input array is passed "
                      "through to the output, with the first element incremented.";
    /* ----- inputs ------- */
    ns->inputs.add("input_u", InputSpec("UInt32 Data",
                                       NTA_BasicType_UInt32, // type
                                       0,                    // count
                                       false,                // required
                                       true,                 // isRegionLevel,
                                       true                  // isDefaultInput
                                       ));

    /* ----- outputs ------ */
    ns->outputs.add("output_u", OutputSpec("UInt32 Data",
                                         NTA_BasicType_UInt32, // type
                                         0,                    // count is dynamic
                                         true,                 // isRegionLevel
                                         true                  // isDefaultOutput
                                         ));
    /* ---- executeCommand ---- */
    ns->commands.add("HelloWorld",  CommandSpec("Hello world command"));

    return ns;
  }

  bool operator==(const RegionImpl &other) const override { return ((PassthruRegion &)other).param == param; }
  inline bool operator!=(const PassthruRegion &other) const { return !operator==(other); }

private:
  int param;
};

} // namespace htm
namespace testing {


TEST(NetworkTest, Scenario2) {
  // This is intended to demonstrate execution order as A B C D C D C D E
  // This time, actually pass data through the links.
  // The dimension defined in the link INPUT.begin should propogate to all regions.
  Network n;
  n.registerRegion("PassthruRegion", new RegisteredRegionImplCpp<PassthruRegion>());

  std::string config = R"(
   {network: [
       {addRegion: {name: "A", type: "PassthruRegion", phase: 1}},
       {addRegion: {name: "B", type: "PassthruRegion", phase: 1}},
       {addRegion: {name: "C", type: "PassthruRegion", phase: 2}},
       {addRegion: {name: "D", type: "PassthruRegion", phase: 2}},
       {addRegion: {name: "E", type: "PassthruRegion", phase: 3}},
       {addLink:   {src: "INPUT.begin", dest: "A.input_u", dim: [10]}},
       {addLink:   {src: "A.output_u", dest: "B.input_u"}},
       {addLink:   {src: "B.output_u", dest: "C.input_u"}},
       {addLink:   {src: "C.output_u", dest: "D.input_u"}},
       {addLink:   {src: "D.output_u", dest: "C.input_u", mode: overwrite }},
       {addLink:   {src: "D.output_u", dest: "E.input_u"}},
    ]})";
  n.configure(config);

//LogLevel ll = NTA_LOG_LEVEL; // this is a global
//n.setLogLevel(LogLevel::LogLevel_Verbose);
  n.initialize();  // initalizes regions and links
//n.setLogLevel(ll);

  // we start with the first element being 0.
  std::vector<UInt32> initialdata = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
  n.setInputData("begin", initialdata);

  // confirm that the INPUT.begin buffer is populated.
  std::shared_ptr<Region> input = n.getRegion("INPUT");
  const Array &input_data = input->getOutputData("begin");
  EXPECT_EQ(input_data.asVector<UInt32>(), initialdata);
  n.run(1, 0); // execute the internal INPUT region.
  n.run(1, 1); // execute A, B once
  n.run(3, 2); // execute C, D three times
  n.run(1, 3); // execute E once

  // We should end with the first element incremented the same number of times
  // that LinkRegion was executed. This shows that the message went through
  // all instances.
  std::shared_ptr<Region> regionE = n.getRegion("E");
  const Array &result = regionE->getOutputData("output_u");
  std::vector<UInt32> expecteddata = {9u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
  EXPECT_EQ(result.asVector<UInt32>(), expecteddata);
}

TEST(NetworkTest, SaveRestore) {
  // Note: this sort-of mimics test in network_test.py "testNetworkPickle"
  Network network;
  network.registerRegion("PassthruRegion", new RegisteredRegionImplCpp<PassthruRegion>());
  auto r_from = network.addRegion("A", "PassthruRegion", "");
  auto r_to = network.addRegion("B", "PassthruRegion", "");

  network.link("INPUT", "A", "", "{dim: [10]}", "begin", "input_u");
  network.link("A", "B", "", "", "output_u", "input_u");
  network.initialize();

  std::vector<UInt32> initialdata = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
  network.setInputData("begin", initialdata);

//LogLevel ll = NTA_LOG_LEVEL; // this is a global
//network.setLogLevel(LogLevel::LogLevel_Verbose);
  network.run(1);
//network.setLogLevel(ll);

  // Confirm we have a reasonable output buffer after one run.
  std::shared_ptr<Region> regionB1 = network.getRegion("B");
  const Array &result1 = regionB1->getOutputData("output_u");
  std::vector<UInt32> expecteddata = {2u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u};
  EXPECT_EQ(result1.asVector<UInt32>(), expecteddata);

  std::stringstream ss;
  network.save(ss);

  Network network2;
  network2.load(ss);

  // Confirm the output buffer is still correct.
  std::shared_ptr<Region> regionB2 = network2.getRegion("B");
  const Array &result2 = regionB2->getOutputData("output_u");
  EXPECT_EQ(result2.asVector<UInt32>(), expecteddata);


  std::string s1 = network.getRegion("B")->executeCommand({"HelloWorld", "26", "64"});
  std::string s2 = network2.getRegion("B")->executeCommand({"HelloWorld", "26", "64"});
  ASSERT_STREQ(s1.c_str(), "Hello World says: arg1=26 arg2=64");
  ASSERT_STREQ(s1.c_str(), s2.c_str());
}

TEST(NetworkTest, Configure) {
  // tests the configure() method with a phase specification.
  std::string config = R"(
   {network: [
       {addRegion: {name: "encoder", type: "RDSEEncoderRegion", params: {size: 1000, sparsity: 0.2, radius: 0.03, seed: 2019, noise: 0.01}, phase: [1,2]}},
       {addRegion: {name: "sp", type: "SPRegion", params: {columnCount: 2048, globalInhibition: true}, phase: [1]}},
       {addRegion: {name: "tm", type: "TMRegion", params: {cellsPerColumn: 8, orColumnOutputs: true}, phase: [1]}},
       {addLink:   {src: "encoder.encoded", dest: "sp.bottomUpIn"}},
       {addLink:   {src: "sp.bottomUpOut", dest: "tm.bottomUpIn"}}
    ]})";
  
  Network net;
  net.configure(config);
  auto r = net.getRegion("encoder");
  ASSERT_STREQ(r->getName().c_str(), "encoder");
  ASSERT_STREQ(r->getType().c_str(), "RDSEEncoderRegion");
  ASSERT_EQ(r->getParameterUInt32("activeBits"), 200u);
  
  std::string name = net.phasesToString();
  ASSERT_STREQ(name.c_str(), "{minEnabledPhase_: 1, maxEnabledPhase_: 2, phases: [[][encoder, sp, tm, ][encoder, ]]}");

  // Note: There is more testing of configuration using the config string 
  //       in the RESTapiTest class.
}



} // namespace testing
