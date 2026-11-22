#include <iostream>
#include <cstring>

class PacketParser {
public:
    void parse(const char* src) {
        char* temp = new char[256];
        strcpy(temp, src);
        std::cout << temp << std::endl;
    }
};
