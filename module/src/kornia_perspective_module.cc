#include "kornia_perspective_module.h"

namespace module{

    int64_t KorniaPerspectiveLayer::deserializeModule(const char* data_p, int64_t start_byte){
        return 0;
    }
    int64_t KorniaPerspectiveLayer::serializeModule(const char* data_p, int64_t start_byte){
        return 0;
    }
    bool KorniaPerspectiveLayer::init(){
        return true;
    }
    bool KorniaPerspectiveLayer::enqueueModule(){
        return true;
    }
    bool KorniaPerspectiveLayer::executeModule(){
        return true;
    }

    bool KorniaPerspectiveLayer::registed_kornia_perspective_layer_ = [](){
        std::string type_name = "KorniaPerspectiveLayer";
        ModuleFactory::registModuleCreator(type_name, 
                                            [](){return BaseModulePtr(std::make_shared<KorniaPerspectiveLayer>());});
        return true;
    }();
}