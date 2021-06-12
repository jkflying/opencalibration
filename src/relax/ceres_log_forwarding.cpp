// clang-format off
#include <glog/logging.h>
#include <glog/log_severity.h>
// clang-format on
#include <spdlog/spdlog.h>

namespace
{

class SpdLogSink : public google::LogSink
{
  public:
    void send(google::LogSeverity severity, const char *full_filename, const char *base_filename, int line,
              const struct ::tm *tm_time, const char *message, size_t message_len) override
    {
        // unused, maybe integrate them somehow?
        (void)full_filename;
        (void)tm_time;

        spdlog::level::level_enum level;
        switch (severity)
        {
        case google::GLOG_INFO:
            level = spdlog::level::debug;
            break;
        case google::GLOG_WARNING:
            level = spdlog::level::info;
            break;
        case google::GLOG_ERROR:
            level = spdlog::level::warn;
            break;
        case google::GLOG_FATAL:
            level = spdlog::level::err;
            break;
        default:
            level = spdlog::level::critical;
        }
        spdlog::log(level, "{}:{}  {}", base_filename, line, std::string_view(message, message_len));
    }
};

struct RegisterCeresLogger
{
    SpdLogSink sink;

    RegisterCeresLogger(const char *name)
    {
        google::InitGoogleLogging(name);
        google::AddLogSink(&sink);
    }
};

static RegisterCeresLogger *__init_at_load = new RegisterCeresLogger("opencalibration");
} // namespace
