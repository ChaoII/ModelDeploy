// ModelDeploy demo: 文档理解（DocToMarkdown）。
// 加载 layout 布局模型（必选）+ 公式识别 FormulaRecognizer / OCR PaddleOCR / 表格 PPStructureV2Table（至少其一）
// -> 对整页图做版面分析 -> 按区域输出 Markdown（公式用 $...$，表格嵌 HTML）到 stdout。
//
// 用法：
//   demo_doc <layout.onnx> <image> [<formula.onnx> [<char_dict>]] \
//       [--ocr <det> <cls> <rec> <dict>] [--table <det> <rec> <table> <rec_label> <table_char_dict>]
// 示例：
//   demo_doc layout.onnx page.jpg formula.onnx formula_dict.txt \
//       --ocr det.onnx cls.onnx rec.onnx ppocrv4_dict.txt \
//       --table det.onnx rec.onnx table.onnx rec_label.txt table_dict.txt
#include "vision/ocr/doc_to_markdown.h"
#include "vision/ocr/formula_recognition.h"
#include "vision/ocr/structurev2_layout.h"
#include "vision/ocr/ppstructurev2_table.h"
#include "vision/ocr/ppocr.h"
#include "vision/common/image_data.h"
#include "runtime/runtime_option.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace ocr = modeldeploy::vision::ocr;

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <layout.onnx> <image> [<formula.onnx> [<char_dict>]]\n"
        "       [--ocr <det.onnx> <cls.onnx> <rec.onnx> <dict.txt>]\n"
        "       [--table <det.onnx> <rec.onnx> <table.onnx> <rec_label.txt> <table_char_dict.txt>]\n"
        "\n"
        "把整页图按版面区域识别为 Markdown 打印到 stdout。\n"
        "layout 必选；formula/ocr/table 至少提供一个内容识别器。\n",
        argv0);
}

int main(int argc, char** argv) {
    // ---- 参数解析（手工）----
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() < 2) { usage(argv[0]); return 1; }
    const std::string layout_file = args[0];
    const std::string image_file = args[1];

    std::string formula_file, formula_dict;
    std::vector<std::string> ocr_args, table_args;
    size_t i = 2;
    // 可选 formula: <formula.onnx> [<char_dict>]（以 --ocr / --table 为界）
    if (i < args.size() && args[i] != "--ocr" && args[i] != "--table") {
        formula_file = args[i++];
        if (i < args.size() && args[i] != "--ocr" && args[i] != "--table")
            formula_dict = args[i++];
    }
    for (; i < args.size(); ++i) {
        if (args[i] == "--ocr") {
            while (i + 1 < args.size() && args[i + 1] != "--table")
                ocr_args.push_back(args[++i]);
        } else if (args[i] == "--table") {
            while (i + 1 < args.size())
                table_args.push_back(args[++i]);
        } else {
            std::fprintf(stderr, "unexpected argument: %s\n", args[i].c_str());
            usage(argv[0]);
            return 1;
        }
    }
    if (ocr_args.size() != 0 && ocr_args.size() != 4) {
        std::fprintf(stderr, "--ocr 需要 4 个参数: <det> <cls> <rec> <dict>\n");
        return 1;
    }
    if (table_args.size() != 0 && table_args.size() != 5) {
        std::fprintf(stderr, "--table 需要 5 个参数: <det> <rec> <table> <rec_label> <table_char_dict>\n");
        return 1;
    }
    if (formula_file.empty() && ocr_args.empty() && table_args.empty()) {
        std::fprintf(stderr, "错误：需要提供至少一个内容识别器（--formula / --ocr / --table）\n");
        usage(argv[0]);
        return 1;
    }

    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();

    // ---- 1. 版面分析（必须）----
    auto layout = std::make_unique<ocr::StructureV2Layout>(layout_file, opt);
    if (!layout->is_initialized()) {
        std::fprintf(stderr, "错误：加载 layout 模型失败（检查文件路径是否有效）：%s\n", layout_file.c_str());
        return 1;
    }

    // ---- 2. 内容识别器（公式 / OCR / 表格，可组合）----
    std::unique_ptr<ocr::FormulaRecognizer> formula;
    if (!formula_file.empty()) {
        formula = std::make_unique<ocr::FormulaRecognizer>(formula_file, formula_dict, opt);
        if (!formula->is_initialized()) {
            std::fprintf(stderr, "错误：加载公式识别模型失败：%s\n", formula_file.c_str());
            return 1;
        }
    }
    std::unique_ptr<ocr::PaddleOCR> paddle_ocr;
    if (ocr_args.size() == 4) {
        paddle_ocr = std::make_unique<ocr::PaddleOCR>(
            ocr_args[0], ocr_args[1], ocr_args[2], ocr_args[3], opt);
        if (!paddle_ocr->is_initialized()) {
            std::fprintf(stderr, "错误：加载 OCR pipeline 失败\n");
            return 1;
        }
    }
    std::unique_ptr<ocr::PPStructureV2Table> table;
    if (table_args.size() == 5) {
        table = std::make_unique<ocr::PPStructureV2Table>(
            table_args[0], table_args[1], table_args[2], table_args[3], table_args[4],
            960, 0.3, 0.6, 1.5, "slow", false, 8, opt);
        if (!table->is_initialized()) {
            std::fprintf(stderr, "错误：加载表格模型失败\n");
            return 1;
        }
    }

    // ---- 3. 装配 DocToMarkdown（owning unique_ptr 重载）----
    ocr::DocToMarkdown dm;
    dm.set_layout(std::move(layout));
    if (formula)     dm.set_formula(std::move(formula));
    if (paddle_ocr)  dm.set_ocr(std::move(paddle_ocr));
    if (table)       dm.set_table(std::move(table));
    if (!dm.ready()) {
        std::fprintf(stderr, "错误：DocToMarkdown 未就绪（缺少 layout 或至少一个内容识别器）\n");
        return 1;
    }

    // ---- 4. 读图并推理 ----
    auto page = modeldeploy::vision::ImageData::imread(image_file);
    if (page.empty()) {
        std::fprintf(stderr, "错误：无法读取图片：%s\n", image_file.c_str());
        return 1;
    }
    std::string markdown;
    if (!dm.predict(page, &markdown)) {
        std::fprintf(stderr, "错误：DocToMarkdown.predict 失败\n");
        return 1;
    }
    std::fprintf(stdout, "%s\n", markdown.c_str());
    std::fprintf(stderr, "已完成，Markdown 共 %zu 字节\n", markdown.size());
    return 0;
}
