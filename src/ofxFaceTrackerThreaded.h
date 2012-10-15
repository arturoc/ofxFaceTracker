#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxFaceTracker.h"

class ofxFaceTrackerThreaded : public ofThread, public ofxFaceTracker {
public:
	ofxFaceTrackerThreaded()
	:threadedIfFound(false)
	,failedMiddle(true)
	,meanObjectPointsReady(false)
	,newFrameAnalized(false)
	,scale(1)
	{
	}
	~ofxFaceTrackerThreaded() {
		newFrameCondition.signal();
		waitForThread(true);
	}
	void setup() {
		failed = true;
		tracker.setup();
		startThread();
	}
	bool update(cv::Mat image) {
		if((failed || threadedIfFound) && dataMutex.tryLock()){
			image.copyTo(imageMiddle);
			newFrameCondition.signal();
			updateData();
			dataMutex.unlock();
		}else if(!failed){
			dataMutex.lock();
			image.copyTo(imageMiddle);
			analyzeData();
			updateData();
			dataMutex.unlock();
		}
		return !failed;
	}
	const cv::Mat& getObjectPointsMat() const {
		return objectPointsMatFront;
	}
	ofVec2f getImagePoint(int i) const {
		if(failed) {
			return ofVec2f();
		}
		return imagePointsFront[i];
	}
	ofVec3f getObjectPoint(int i) const {
		if(failed) {
			return ofVec3f();
		}
		return objectPointsFront[i];
	}
	ofVec3f getMeanObjectPoint(int i) const {
		if(meanObjectPointsReady) {
			return meanObjectPointsFront[i];
		} else {
			return ofVec3f();
		}
	}
	bool getVisibility(int i) const {
		return failed;
	}
	ofVec3f getOrientation() const {
		return orientation;
	}
	ofVec2f getPosition() const {
		return position;
	}
	float getScale() const {
		return scale;
	}
	int size() const{
		return objectPointsFront.size();
	}
	
	ofxFaceTracker tracker;
	bool threadedIfFound;

protected:
	void threadedFunction() {
		dataMutex.lock();
		while(isThreadRunning()) {
			newFrameCondition.wait(dataMutex);
			analyzeData();
			swap(imageMiddle,imageBack);
		}
	}
	void analyzeData(){
		swap(imageMiddle,imageBack);

		tracker.setRescale(rescale);
		tracker.setIterations(iterations);
		tracker.setClamp(clamp);
		tracker.setTolerance(tolerance);
		tracker.setAttempts(attempts);
		tracker.setUseInvisible(useInvisible);

		tracker.update(imageBack);

		objectPointsMiddle = tracker.getObjectPoints();
		imagePointsMiddle = tracker.getImagePoints();
		meanObjectPointsMiddle = tracker.getMeanObjectPoints();
		failedMiddle = !tracker.getFound();
		position = tracker.getPosition();
		orientation = tracker.getOrientation();
		scale = tracker.getScale();
		objectPointsMatMiddle = tracker.getObjectPointsMat();
		newFrameAnalized = true;
	}
	void updateData(){
		if(newFrameAnalized){
			swap(objectPointsFront,objectPointsMiddle);
			swap(imagePointsFront,imagePointsMiddle);
			swap(meanObjectPointsFront,meanObjectPointsMiddle);
			swap(objectPointsMatFront,objectPointsMatMiddle);
			failed = failedMiddle;
			if(!failed) {
				meanObjectPointsReady = true;
			}
		}
		newFrameAnalized = false;
	}

	ofMutex dataMutex;
	
	cv::Mat imageMiddle, imageBack;
	vector<ofVec3f> objectPointsFront, objectPointsMiddle;
	vector<ofVec2f> imagePointsFront, imagePointsMiddle;
	vector<ofVec3f> meanObjectPointsFront, meanObjectPointsMiddle;
	bool failedMiddle;
	bool meanObjectPointsReady;
	bool newFrameAnalized;
	
	ofVec3f orientation;
	float scale;
	ofVec2f position;
	cv::Mat objectPointsMatBack, objectPointsMatMiddle, objectPointsMatFront; 
	Poco::Condition newFrameCondition,frameAnalized;
};
